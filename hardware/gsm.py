"""Sends SMS alerts through a SIM7600/SIM800L GSM modem.

READING THIS FILE
The modem is a serial device that speaks AT commands. Only one thread may talk to it,
because every command expects a reply and the replies are not tagged. So this class
opens the port, puts the modem into text mode, and then hands all sending to one
background worker fed by a queue.

send() never blocks: the control loop drops a short string in the queue and moves on.
If the queue is full the oldest message is effectively dropped - a missed text must
never delay the motors. A missing modem or SIM is reported once and then ignored; SMS is
the only optional subsystem in the build.
"""
import queue                        # hand messages from the control loop to the worker
import threading                    # the AT-command worker runs on its own thread
import time                         # timeouts and the post-open settle delay

import serial                       # pyserial, same library as the GNSS and LiDAR


class Gsm:
    """One modem, one worker thread, one queue of pending texts."""

    def __init__(self, cfg):
        self.cfg = cfg                                    # port, baud, phone number
        self.number = str(cfg.get("phone", "")).strip()   # destination number
        self.ser = None                                   # serial handle, if it opened
        self.ready = False                                # True after AT+CMGF succeeds
        self.queue = queue.Queue(maxsize=8)               # bounded: never grow forever
        if not cfg.get("enabled", False):                 # switch in config.yaml
            print("[gsm] disabled", flush=True)
            return
        if not self.number:                               # no destination configured
            print("[gsm] no phone number configured", flush=True)
            return
        try:
            self.ser = serial.Serial(cfg["port"], cfg["baud"], timeout=1)
            time.sleep(0.5)                               # most modems reboot on DTR
            reply = b""
            for _ in range(5):                            # give a cold modem time to boot
                reply = self._cmd("AT", (b"OK",), 1.0)
                if b"OK" in reply:                        # it is alive
                    break
                time.sleep(0.5)
            if b"OK" not in reply:                        # never answered
                raise RuntimeError("modem did not answer AT")
            if b"OK" not in self._cmd("AT+CMGF=1", (b"OK",), 2.0):   # text mode
                raise RuntimeError("AT+CMGF=1 rejected")
            self.ready = True
            threading.Thread(target=self._run, daemon=True).start()
            print(f"[gsm] ready on {cfg['port']}", flush=True)
        except Exception as exc:                          # missing port, no SIM, no power
            self.ser = None                               # stay disabled
            self.ready = False
            print(f"[gsm] unavailable: {exc}", flush=True)

    def _read_until(self, tokens, timeout):
        """Read bytes until one of `tokens` appears or the deadline passes."""
        end = time.monotonic() + timeout               # absolute deadline
        buf = b""                                      # everything received so far
        while time.monotonic() < end:                  # keep reading
            chunk = self.ser.read(1)                   # one byte (port timeout is 1 s)
            if not chunk:                              # nothing yet
                continue                               # check the clock and retry
            buf += chunk                               # accumulate
            if any(token in buf for token in tokens):  # expected reply arrived
                break                                  # stop waiting
        return buf                                     # caller inspects it

    def _cmd(self, command, tokens, timeout):
        """Send one AT command and collect the reply."""
        self.ser.reset_input_buffer()                  # drop stale `+CREG` notices
        self.ser.write((command + "\r").encode("ascii"))
        return self._read_until(tokens, timeout)

    def send(self, text):
        """Queue one text. Returns False if SMS is unavailable or the queue is full."""
        if self.ser is None or not self.ready:         # no modem
            return False                               # caller carries on
        try:
            self.queue.put_nowait(text)                # never block the control loop
            return True                                # accepted
        except queue.Full:                             # modem far behind
            return False                               # drop this alert

    def _run(self):
        """Worker loop: one SMS at a time, forever."""
        while True:                                    # lives as long as the process
            text = self.queue.get()                    # wait for the next alert
            try:
                self._send_now(text)                   # slow part happens here
            except Exception as exc:                   # unplugged, SIM removed, etc.
                print(f"[gsm] send failed: {exc}", flush=True)

    def _send_now(self, text):
        """The actual AT+CMGS sequence. Raises if the modem rejects the message."""
        self.ser.reset_input_buffer()                  # clear unsolicited messages
        self.ser.write(f'AT+CMGS="{self.number}"\r'.encode("ascii"))
        prompt = self._read_until((b">",), 5.0)        # modem answers with "> "
        if b">" not in prompt:                         # number rejected or not registered
            raise RuntimeError("no SMS prompt (SIM registered?)")
        body = text.encode("ascii", "ignore") + b"\x1a"   # Ctrl-Z sends the message
        self.ser.write(body)                           # hand over the text
        reply = self._read_until((b"+CMGS", b"ERROR"), 30.0)
        if b"+CMGS" not in reply:                      # no reference number came back
            raise RuntimeError("modem did not confirm the SMS")

    def close(self):
        """Stop using the port. The daemon thread exits with the process."""
        self.ready = False                             # stop accepting new messages
        if self.ser is not None:                       # only if it opened
            try:
                self.ser.close()                       # release the USB port
            except Exception:
                pass                                   # shutdown must never raise
            self.ser = None                            # send() now returns False
