"""Output GPIO Raspberry Pi: pin terpisah untuk hasil OK dan NG.

Di PC Windows tidak ada GPIO. Aturan tetap disimpan dan baru menggerakkan pin
ketika aplikasi berjalan di Raspberry Pi (gpiozero, lgpio, atau RPi.GPIO).
"""
import importlib

from .qt import QtCore


def _backend_name():
    for name, mod in (("gpiozero", "gpiozero"), ("lgpio", "lgpio"), ("rpigpio", "RPi.GPIO")):
        try:
            importlib.import_module(mod)
            return name
        except Exception:
            continue
    return None


class GpioChannel(QtCore.QObject):
    """Satu pin BCM yang menyala saat hasil tertentu (OK atau NG) aktif."""

    def __init__(self, parent, label):
        super().__init__(parent)
        self.label = label
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_pulse_end)
        self._hw = None
        self._enabled = False
        self._pin = 17
        self._active_high = True
        self._pulse_ms = 500
        self._ready = False
        self._on = False
        self._latched = False
        self._backend = None
        self._dev = None
        self._uses_lgpio = False
        self._gpio = None
        self._tone = ""
        self._message = f"Aturan {label} mati."

    def status(self):
        return self._tone, self._message

    def configure(self, enabled, pin, active_high, pulse_ms):
        """Simpan target pin tanpa membuka hardware. Mengembalikan True bila target berubah."""
        enabled = bool(enabled)
        pin = int(pin)
        active_high = bool(active_high)
        self._pulse_ms = max(0, int(pulse_ms))
        hw = (enabled, pin, active_high)
        changed = hw != self._hw
        self._hw = hw
        self._enabled = enabled
        self._pin = pin
        self._active_high = active_high
        if not changed:
            return False
        self._release()
        self._latched = False
        if not enabled:
            self._set_status("", f"Aturan {self.label} mati. Pin tidak diklaim.")
        return True

    def ensure_open(self):
        if not self._enabled:
            return
        if self._ready:
            self._set_status("ok", self._ready_text())
            return
        taken = self._other_ready_pin()
        if taken == self._pin:
            self._set_status("ng", f"Pin BCM {self._pin} sudah dipakai hasil lain. Pilih pin yang berbeda.")
            return
        self._open(self._pin, self._active_high)

    def set_active(self, active: bool):
        if not self._enabled or not self._ready:
            return
        active = bool(active)
        if self._pulse_ms <= 0:
            self._latched = active
            self._timer.stop()
            self._write(active)
            return
        if active:
            if not self._latched:
                self._latched = True
                self._write(True, force=True)
                self._timer.start(self._pulse_ms)
        elif self._latched or self._on:
            self._latched = False
            self._timer.stop()
            self._write(False)

    def test(self):
        if not self._enabled:
            self._set_status("warn", f"Aktifkan aturan GPIO {self.label} terlebih dahulu.")
            return
        if not self._ready:
            return
        ms = self._pulse_ms if self._pulse_ms > 0 else 500
        self._timer.stop()
        self._write(True, force=True)
        self._timer.start(ms)
        self._set_status("ok", f"Uji nyala {self.label}: BCM {self._pin} selama {ms} ms.")

    def close(self):
        self._timer.stop()
        self._release()
        self._hw = None
        self._latched = False

    # ------------------------------------------------------------------
    def _ready_text(self):
        level = "aktif HIGH" if self._active_high else "aktif LOW"
        if self._pulse_ms > 0:
            how = f"pulsa {self._pulse_ms} ms"
        else:
            how = f"menyala selama hasil {self.label}"
        state = " · sedang menyala" if self._on else ""
        return f"{self.label} siap: BCM {self._pin} · {level} · {how}{state}."

    def _set_status(self, tone, message):
        self._tone = tone
        self._message = message

    def _other_ready_pin(self):
        bus = self.parent()
        for ch in (bus.ok, bus.ng):
            if ch is not self and ch._ready:
                return ch._pin
        return None

    def _open(self, pin, active_high):
        self._pin = pin
        self._active_high = active_high
        backend = _backend_name()
        if backend is None:
            self._set_status(
                "warn",
                "GPIO tidak tersedia di komputer ini. Aturan disimpan dan aktif di Raspberry Pi.")
            return
        try:
            if backend == "gpiozero":
                from gpiozero import DigitalOutputDevice
                self._dev = DigitalOutputDevice(pin, active_high=active_high, initial_value=False)
            elif backend == "lgpio":
                lgpio, chip = self.parent().acquire_lgpio()
                try:
                    lgpio.gpio_claim_output(chip, pin, 0 if active_high else 1)
                except Exception:
                    self.parent().release_lgpio()
                    raise
                self._uses_lgpio = True
            else:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW if active_high else GPIO.HIGH)
                self._gpio = GPIO
        except Exception as e:
            self._release()
            self._set_status("ng", f"Gagal membuka pin {self.label} BCM {pin}: {e}")
            return
        self._backend = backend
        self._ready = True
        self._on = False
        self._set_status("ok", self._ready_text())

    def _level(self, on: bool) -> int:
        return 1 if bool(on) == self._active_high else 0

    def _write(self, on: bool, force=False):
        if not self._ready:
            return
        on = bool(on)
        if not force and on == self._on:
            return
        try:
            if self._backend == "gpiozero":
                if on:
                    self._dev.on()
                else:
                    self._dev.off()
            elif self._backend == "lgpio":
                lgpio, chip = self.parent()._lgpio, self.parent()._chip
                lgpio.gpio_write(chip, self._pin, self._level(on))
            elif self._backend == "rpigpio":
                self._gpio.output(self._pin, self._gpio.HIGH if self._level(on) else self._gpio.LOW)
            else:
                return
        except Exception as e:
            self._ready = False
            self._set_status("ng", f"Gagal menulis pin {self.label} BCM {self._pin}: {e}")
            return
        self._on = on
        if self._ready:
            self._set_status("ok", self._ready_text())

    def _on_pulse_end(self):
        self._write(False)

    def _release(self):
        self._timer.stop()
        if self._ready:
            try:
                self._write(False, force=True)
            except Exception:
                pass
        self._ready = False
        self._on = False
        dev, self._dev = self._dev, None
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass
        if self._uses_lgpio:
            bus = self.parent()
            try:
                if bus._lgpio is not None and bus._chip is not None:
                    bus._lgpio.gpio_free(bus._chip, self._pin)
            except Exception:
                pass
            bus.release_lgpio()
            self._uses_lgpio = False
        if self._gpio is not None:
            try:
                self._gpio.cleanup(self._pin)
            except Exception:
                pass
        self._gpio = None
        self._backend = None


class GpioOutputs(QtCore.QObject):
    """Dua output: pin OK dan pin NG."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ok = GpioChannel(self, "OK")
        self.ng = GpioChannel(self, "NG")
        self._lgpio = None
        self._chip = None
        self._chip_users = 0

    def apply_config(self, cfg):
        self.ok.configure(cfg.gpio_ok_enabled, cfg.gpio_ok_pin, cfg.gpio_ok_active_high, cfg.gpio_ok_pulse_ms)
        self.ng.configure(cfg.gpio_ng_enabled, cfg.gpio_ng_pin, cfg.gpio_ng_active_high, cfg.gpio_ng_pulse_ms)
        if self.ok._enabled and self.ng._enabled and self.ok._pin == self.ng._pin:
            self.ng._release()
            self.ng._set_status("ng", f"Pin BCM {self.ng._pin} sudah dipakai hasil OK. Pilih pin yang berbeda.")
            self.ok.ensure_open()
            return
        self.ok.ensure_open()
        self.ng.ensure_open()

    def set_result(self, judgement):
        """OK menyalakan pin OK. NG dan lubang tidak terdeteksi menyalakan pin NG."""
        if judgement is True:
            judgement = "OK"
        elif judgement is False or judgement is None:
            judgement = ""
        else:
            judgement = str(judgement)
        self.ok.set_active(judgement == "OK")
        self.ng.set_active(judgement in ("NG", "NO_HOLE"))

    def test(self, kind):
        (self.ok if kind == "ok" else self.ng).test()

    def close(self):
        self.ok.close()
        self.ng.close()

    def acquire_lgpio(self):
        if self._chip is None:
            import lgpio
            self._lgpio = lgpio
            self._chip = lgpio.gpiochip_open(0)
        self._chip_users += 1
        return self._lgpio, self._chip

    def release_lgpio(self):
        self._chip_users = max(0, self._chip_users - 1)
        if self._chip_users == 0 and self._chip is not None:
            try:
                self._lgpio.gpiochip_close(self._chip)
            except Exception:
                pass
            self._chip = None
            self._lgpio = None
