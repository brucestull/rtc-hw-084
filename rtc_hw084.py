"""Read and write the time on an HW-084 (DS3231) real-time clock module.

The HW-084 is a DS3231 I2C real-time clock carrier board with a CR2032
battery holder and an on-board AT24C32 EEPROM. It lets ``CLICK-CLOCK``
(a Raspberry Pi Zero) keep accurate time while it has no network
connection and therefore no NTP.

Wiring
------
The HW-084 connects to the Raspberry Pi Zero's 40-pin header::

    HW-084      Raspberry Pi Zero
    ------      -----------------------------------------
    VCC    ->   3V3     (BCM --, physical pin 1)
    GND    ->   GND     (BCM --, physical pin 6)
    SDA    ->   GPIO 2  (BCM  2, physical pin 3)
    SCL    ->   GPIO 3  (BCM  3, physical pin 5)
    SQW    ->   optional: any free GPIO, for the alarm interrupt or
                the 1 Hz square wave
    32K    ->   optional: 32.768 kHz output, normally left unconnected

The DS3231 is 3.3 V tolerant, so power it from 3V3 rather than 5V to
keep the I2C lines at a level the Pi is happy with.

Prerequisites
-------------
Enable the I2C bus on the Pi (``sudo raspi-config`` -> Interface Options
-> I2C, then reboot) and install the dependency::

    uv add smbus2

Confirm the module is visible before using this module::

    sudo apt install -y i2c-tools
    i2cdetect -y 1

The DS3231 should appear at ``0x68`` and the EEPROM at ``0x57``.

Time zones
----------
The DS3231 has no concept of a time zone: it stores a bare wall-clock
date and time. This module stores UTC by default (see ``STORE_UTC``) so
that daylight-saving transitions can never corrupt the stored value.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from smbus2 import SMBus, i2c_msg

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Hardware configuration
#
# These are module-level so they can be adjusted without touching the
# logic below.
# --------------------------------------------------------------------------

#: I2C address of the DS3231 real-time clock on the HW-084 board.
#: Fixed at 0x68 by the DS3231 silicon; exposed here in case the module is
#: swapped for a different part.
RTC_I2C_ADDRESS = 0x68

#: I2C address of the AT24C32 EEPROM on the same board. Unused by this
#: module, but recorded here so the two addresses stay together. Selectable
#: to 0x50-0x57 with the A0/A1/A2 solder pads; 0x57 is the factory default.
EEPROM_I2C_ADDRESS = 0x57

#: I2C bus number. Bus 1 is the 40-pin header on every Pi since the Model B
#: Rev 2, the Pi Zero included.
I2C_BUS_NUMBER = 1

# --- Pin assignments (BCM numbering) --------------------------------------

#: Data line. Hardware I2C on bus 1 is fixed to this pin.
SDA_PIN = 2

#: Clock line. Hardware I2C on bus 1 is fixed to this pin.
SCL_PIN = 3

#: GPIO wired to the HW-084 SQW pin, or ``None`` when it is not connected.
#: Set this when wiring up the alarm interrupt or the 1 Hz square wave.
SQW_PIN = None

#: GPIO wired to the HW-084 32K pin, or ``None`` when it is not connected.
THIRTY_TWO_KHZ_PIN = None

# --- Behaviour ------------------------------------------------------------

#: Store UTC in the RTC rather than local time. Keep this ``True`` unless
#: there is a strong reason not to: local time is ambiguous for one hour
#: every autumn.
STORE_UTC = True

# --------------------------------------------------------------------------
# DS3231 register map
# --------------------------------------------------------------------------

REGISTER_SECONDS = 0x00
REGISTER_MINUTES = 0x01
REGISTER_HOURS = 0x02
REGISTER_DAY_OF_WEEK = 0x03
REGISTER_DATE = 0x04
REGISTER_MONTH = 0x05
REGISTER_YEAR = 0x06
REGISTER_CONTROL = 0x0E
REGISTER_STATUS = 0x0F

#: Number of timekeeping registers read or written as one block.
TIMEKEEPING_REGISTER_COUNT = 7

# Bit masks within the timekeeping registers.
MASK_SECONDS = 0b0111_1111
MASK_MINUTES = 0b0111_1111
MASK_HOURS_24 = 0b0011_1111
MASK_HOURS_12 = 0b0001_1111
MASK_DATE = 0b0011_1111
MASK_MONTH = 0b0001_1111

BIT_12_HOUR_MODE = 0b0100_0000  # Hours register: 1 = 12-hour mode.
BIT_PM = 0b0010_0000  # Hours register: PM flag, 12-hour mode only.
BIT_CENTURY = 0b1000_0000  # Month register: 1 = year 2100-2199.
BIT_OSCILLATOR_STOPPED = 0b1000_0000  # Status register: OSF.

#: Lowest and highest years the DS3231 can represent, given the two-digit
#: year register plus the century bit.
MINIMUM_YEAR = 2000
MAXIMUM_YEAR = 2199


class RTCError(Exception):
    """Base class for errors raised by this module."""


class RTCTimeInvalidError(RTCError):
    """The RTC's oscillator stopped, so its stored time is not trustworthy.

    The DS3231 raises its oscillator-stop flag when it loses both VCC and
    battery power, or when the coin cell is dead or missing. The time read
    back after such an event is whatever the counters happened to land on,
    so it must not be used to set the system clock.

    Call :func:`set_time` to write a known-good time; that clears the flag.
    """


class RTCValueError(RTCError, ValueError):
    """A value is outside the range the DS3231 can store."""


# --------------------------------------------------------------------------
# Binary-coded decimal helpers
#
# The DS3231 stores every field as BCD: each nibble holds one decimal
# digit, so decimal 47 is stored as 0x47.
# --------------------------------------------------------------------------


def _bcd_to_int(value: int) -> int:
    """Convert a binary-coded decimal byte to an ``int``.

    Args:
        value: A BCD-encoded byte, for example ``0x47``.

    Returns:
        The decimal value, for example ``47``.
    """
    return (value >> 4) * 10 + (value & 0x0F)


def _int_to_bcd(value: int) -> int:
    """Convert an ``int`` in the range 0-99 to a binary-coded decimal byte.

    Args:
        value: A decimal value between 0 and 99 inclusive.

    Returns:
        The BCD-encoded byte, for example ``0x47`` for ``47``.

    Raises:
        RTCValueError: If ``value`` cannot be held in two BCD digits.
    """
    if not 0 <= value <= 99:
        raise RTCValueError(f"{value} cannot be encoded as two BCD digits")
    return ((value // 10) << 4) | (value % 10)


def _decode_hours(register_value: int) -> int:
    """Decode the DS3231 hours register into a 0-23 hour.

    Handles both the 12-hour and 24-hour representations, since the module
    may have been set by other software before this one ran.

    Args:
        register_value: The raw byte read from :data:`REGISTER_HOURS`.

    Returns:
        The hour of the day, 0-23.
    """
    if register_value & BIT_12_HOUR_MODE:
        hour = _bcd_to_int(register_value & MASK_HOURS_12)
        is_pm = bool(register_value & BIT_PM)
        if hour == 12:
            # 12 AM is midnight (0); 12 PM is noon (12).
            return 12 if is_pm else 0
        return hour + 12 if is_pm else hour
    return _bcd_to_int(register_value & MASK_HOURS_24)


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------


def get_time(
    *,
    bus_number: int = I2C_BUS_NUMBER,
    address: int = RTC_I2C_ADDRESS,
    require_valid: bool = True,
) -> datetime:
    """Read the current date and time from the HW-084 RTC module.

    Args:
        bus_number: I2C bus to read from. Defaults to
            :data:`I2C_BUS_NUMBER`.
        address: I2C address of the DS3231. Defaults to
            :data:`RTC_I2C_ADDRESS`.
        require_valid: When ``True``, refuse to return a time whose
            oscillator-stop flag is set. Pass ``False`` only when
            inspecting a module whose battery is known to have failed.

    Returns:
        The time held by the RTC. Timezone-aware and in UTC when
        :data:`STORE_UTC` is ``True``, otherwise naive local time.

    Raises:
        RTCTimeInvalidError: If the RTC lost power since it was last set
            and ``require_valid`` is ``True``.
        OSError: If the I2C bus or the module cannot be reached. A
            ``FileNotFoundError`` usually means I2C is not enabled on the
            Pi; an ``OSError`` with errno 121 means nothing answered at
            ``address``.
    """
    with SMBus(bus_number) as bus:
        status = bus.read_byte_data(address, REGISTER_STATUS)

        # Read all seven timekeeping registers in one transaction so the
        # fields cannot straddle a tick: seconds could roll over to 00
        # between two separate reads, giving an hour that is 60 minutes
        # fast.
        write = i2c_msg.write(address, [REGISTER_SECONDS])
        read = i2c_msg.read(address, TIMEKEEPING_REGISTER_COUNT)
        bus.i2c_rdwr(write, read)
        registers = list(read)

    oscillator_stopped = bool(status & BIT_OSCILLATOR_STOPPED)
    if oscillator_stopped:
        message = (
            "The DS3231 oscillator-stop flag is set: the module lost power "
            "since it was last set, so its time is not trustworthy. Check "
            "the CR2032 cell and call set_time() to reset it."
        )
        if require_valid:
            raise RTCTimeInvalidError(message)
        logger.warning(message)

    second = _bcd_to_int(registers[REGISTER_SECONDS] & MASK_SECONDS)
    minute = _bcd_to_int(registers[REGISTER_MINUTES] & MASK_MINUTES)
    hour = _decode_hours(registers[REGISTER_HOURS])
    day = _bcd_to_int(registers[REGISTER_DATE] & MASK_DATE)
    month_register = registers[REGISTER_MONTH]
    month = _bcd_to_int(month_register & MASK_MONTH)
    century_base = 2100 if month_register & BIT_CENTURY else 2000
    year = century_base + _bcd_to_int(registers[REGISTER_YEAR])

    moment = datetime(year, month, day, hour, minute, second)
    if STORE_UTC:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def set_time(
    moment: datetime | None = None,
    *,
    bus_number: int = I2C_BUS_NUMBER,
    address: int = RTC_I2C_ADDRESS,
) -> datetime:
    """Write a date and time to the HW-084 RTC module.

    Writing the time also clears the DS3231's oscillator-stop flag, so a
    later :func:`get_time` will accept the value.

    Args:
        moment: The time to store. Defaults to now. A timezone-aware value
            is converted to UTC when :data:`STORE_UTC` is ``True``; a naive
            value is written as given, on the assumption that it is already
            in the reference frame the module stores.
        bus_number: I2C bus to write to. Defaults to
            :data:`I2C_BUS_NUMBER`.
        address: I2C address of the DS3231. Defaults to
            :data:`RTC_I2C_ADDRESS`.

    Returns:
        The value actually written to the registers, after any conversion
        to UTC and after discarding sub-second precision.

    Raises:
        RTCValueError: If ``moment`` falls outside the years the DS3231 can
            represent (2000-2199).
        OSError: If the I2C bus or the module cannot be reached.
    """
    if moment is None:
        moment = datetime.now(timezone.utc) if STORE_UTC else datetime.now()

    if STORE_UTC and moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)

    # The DS3231 counts whole seconds only.
    moment = moment.replace(microsecond=0)

    if not MINIMUM_YEAR <= moment.year <= MAXIMUM_YEAR:
        raise RTCValueError(
            f"The DS3231 can only store years {MINIMUM_YEAR}-{MAXIMUM_YEAR}; "
            f"got {moment.year}."
        )

    century_bit = BIT_CENTURY if moment.year >= 2100 else 0
    two_digit_year = moment.year % 100

    registers = [0] * TIMEKEEPING_REGISTER_COUNT
    registers[REGISTER_SECONDS] = _int_to_bcd(moment.second)
    registers[REGISTER_MINUTES] = _int_to_bcd(moment.minute)
    # Bit 6 left clear, which selects 24-hour mode.
    registers[REGISTER_HOURS] = _int_to_bcd(moment.hour)
    # The DS3231's day-of-week register is just a 1-7 counter that the chip
    # rolls over; the meaning is ours to choose. ISO weekdays are used here,
    # so Monday is 1 and Sunday is 7.
    registers[REGISTER_DAY_OF_WEEK] = _int_to_bcd(moment.isoweekday())
    registers[REGISTER_DATE] = _int_to_bcd(moment.day)
    registers[REGISTER_MONTH] = _int_to_bcd(moment.month) | century_bit
    registers[REGISTER_YEAR] = _int_to_bcd(two_digit_year)

    with SMBus(bus_number) as bus:
        # One transaction again, for the same reason as in get_time().
        bus.i2c_rdwr(i2c_msg.write(address, [REGISTER_SECONDS, *registers]))

        # Clear the oscillator-stop flag now that the stored time is known
        # good. Read-modify-write so the alarm flags and the 32 kHz enable
        # bit in the same register are left alone.
        status = bus.read_byte_data(address, REGISTER_STATUS)
        if status & BIT_OSCILLATOR_STOPPED:
            bus.write_byte_data(
                address,
                REGISTER_STATUS,
                status & ~BIT_OSCILLATOR_STOPPED,
            )

    logger.info("Set the HW-084 RTC to %s.", moment.isoformat())
    return moment
