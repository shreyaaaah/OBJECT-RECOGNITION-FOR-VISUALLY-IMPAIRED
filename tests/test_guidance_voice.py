"""Guidance engine, TTS queue, beeps, STT and hand-guidance behaviour."""
import io
import threading
import time

import numpy as np
import pytest

from fewshot_vi.config import config_from_dict
from fewshot_vi.guidance.engine import (PRIORITY_ANNOUNCE, PRIORITY_BEEP, PRIORITY_FIND, GuidanceEngine)
from fewshot_vi.guidance.hands import reach_instruction
from fewshot_vi.types import Decision, GuidanceEvent, Match, Mode, Quality, Track
from fewshot_vi.voice import beeps, stt, tts

from conftest import FakeClock

FRAME = (480, 640, 3)


def good_q(**kw):
    base = dict(blur=100.0, blurry=False, edge_cut=frozenset(), too_small=False, low_margin=False, score=1.0)
    base.update(kw)
    return Quality(**base)


def trk(tid, bbox, name=None, decision=Decision.ACCEPT, mname=None, quality=None):
    t = Track(track_id=tid, bbox=bbox)
    t.smoothed_name = name
    t.match = Match(tid, mname or name, 0.8, 0.2, decision) if (mname or name) else None
    t.quality = quality or good_q()
    return t


@pytest.fixture
def eng():
    clk = FakeClock(0.0)
    e = GuidanceEngine(config_from_dict(), memory_answer=lambda n: f"Memory: {n} was on the desk.", clock=clk)
    e._clk = clk
    return e


KEYS_BOX = (204, 220, 244, 260)   # 40 px wide, 15% left of centre -> "slightly left", ~1 m


def test_identify_announces_with_direction_and_distance(eng):
    ev = eng.process([trk(1, KEYS_BOX, "keys")], Mode.IDENTIFY, None, FRAME)
    assert [e.text for e in ev] == ["Keys, slightly left, about one metre."]
    assert ev[0].kind == "announce" and ev[0].priority == PRIORITY_ANNOUNCE and ev[0].pan < 0


def test_identify_announce_cooldown_per_object(eng):
    t = [trk(1, KEYS_BOX, "keys")]
    assert len(eng.process(t, Mode.IDENTIFY, None, FRAME)) == 1
    eng._clk.advance(1.0)
    assert eng.process(t, Mode.IDENTIFY, None, FRAME) == []
    other = t + [trk(2, (400, 220, 440, 260), "wallet")]
    ev = eng.process(other, Mode.IDENTIFY, None, FRAME)
    assert len(ev) == 1 and ev[0].text.startswith("Wallet")  # a new object is still announced
    eng._clk.advance(3.0)
    assert len(eng.process(t, Mode.IDENTIFY, None, FRAME)) == 1


def test_unconfirmed_tracks_are_never_named(eng):
    t = trk(1, KEYS_BOX, None, Decision.ACCEPT, mname="keys")  # provisional ACCEPT, not yet smoothed
    ev = eng.process([t], Mode.IDENTIFY, None, FRAME)
    assert all("Keys" not in e.text for e in ev)


def test_random_unknown_object_stays_silent_even_if_blurry(eng):
    t = trk(1, KEYS_BOX, None, Decision.UNKNOWN, mname="keys", quality=good_q(blurry=True))
    assert eng.process([t], Mode.IDENTIFY, None, FRAME) == []


def test_no_unsure_hint_while_lookalike_rerank_is_pending(eng):
    t = trk(1, KEYS_BOX, None, Decision.AMBIGUOUS, mname="keys", quality=good_q(blurry=True))
    t.rerank_pending = True
    assert eng.process([t], Mode.IDENTIFY, None, FRAME) == []
    t.rerank_pending = False
    assert [e.text for e in eng.process([t], Mode.IDENTIFY, None, FRAME)] == ["Hold the camera steady."]


def test_ambiguous_track_triggers_one_hint_with_cooldown(eng):
    t = trk(1, KEYS_BOX, None, Decision.AMBIGUOUS, mname="keys", quality=good_q(blurry=True))
    ev = eng.process([t], Mode.IDENTIFY, None, FRAME)
    assert [e.text for e in ev] == ["Hold the camera steady."] and ev[0].kind == "hint"
    eng._clk.advance(1)
    assert eng.process([t], Mode.IDENTIFY, None, FRAME) == []   # cooldown


def test_find_ignores_other_objects_and_beeps(eng):
    tracks = [trk(1, KEYS_BOX, "keys"), trk(2, (380, 220, 420, 260), "wallet")]
    ev = eng.process(tracks, Mode.FIND, "wallet", FRAME, find_started=0.0)
    said = [e for e in ev if e.kind == "find"]
    beep = [e for e in ev if e.kind == "beep"]
    assert len(said) == 1 and said[0].text.startswith("Wallet") and said[0].priority == PRIORITY_FIND
    assert "Keys" not in said[0].text
    assert len(beep) == 1 and beep[0].priority == PRIORITY_BEEP and beep[0].pan > 0 and 0.0 <= beep[0].urgency <= 1.0


def test_find_beep_urgency_rises_as_object_gets_closer():
    def urgency(width):
        clk = FakeClock(0.0)
        e = GuidanceEngine(config_from_dict(), clock=clk)
        cx = 320
        box = (cx - width / 2, 240 - width / 2, cx + width / 2, 240 + width / 2)
        ev = e.process([trk(1, box, "keys")], Mode.FIND, "keys", FRAME, find_started=0.0)
        return [x for x in ev if x.kind == "beep"][0].urgency
    assert urgency(30) < urgency(100) < urgency(300)


def test_find_beep_interval_shortens_when_close(eng):
    near = trk(1, (200, 140, 440, 340), "keys")  # wide box -> close
    count = 0
    for _ in range(20):
        eng._clk.advance(0.15)
        count += sum(e.kind == "beep" for e in eng.process([near], Mode.FIND, "keys", FRAME, find_started=0.0))
    far_eng = GuidanceEngine(config_from_dict(), clock=FakeClock(0.0))
    far = trk(1, (300, 230, 320, 250), "keys")  # tiny box -> far
    far_count = 0
    for _ in range(20):
        far_eng.clock.advance(0.15)
        far_count += sum(e.kind == "beep" for e in far_eng.process([far], Mode.FIND, "keys", FRAME, find_started=0.0))
    assert count > far_count


def test_find_not_visible_gives_hint_then_memory_once(eng):
    amb = trk(1, KEYS_BOX, None, Decision.AMBIGUOUS, mname="wallet", quality=good_q(too_small=True))
    ev = eng.process([amb], Mode.FIND, "wallet", FRAME, find_started=0.0)
    assert [e.text for e in ev] == ["Move closer."]
    eng._clk.advance(9.0)
    assert not any(e.kind == "memory" for e in eng.process([], Mode.FIND, "wallet", FRAME, find_started=0.0))
    eng._clk.advance(2.0)  # 11 s > find_memory_wait_s (10)
    ev = eng.process([], Mode.FIND, "wallet", FRAME, find_started=0.0)
    assert [e.kind for e in ev] == ["memory"] and ev[0].text == "Memory: wallet was on the desk."
    eng._clk.advance(30.0)
    assert eng.process([], Mode.FIND, "wallet", FRAME, find_started=0.0) == []  # only once per search


def test_find_hint_only_for_the_target(eng):
    other = trk(1, KEYS_BOX, None, Decision.AMBIGUOUS, mname="keys", quality=good_q(blurry=True))
    assert eng.process([other], Mode.FIND, "wallet", FRAME, find_started=0.0) == []


def test_mute_and_enrol_are_silent(eng):
    t = [trk(1, KEYS_BOX, "keys")]
    assert eng.process(t, Mode.IDENTIFY, None, FRAME, muted=True) == []
    assert eng.process(t, Mode.ENROL, None, FRAME) == []


def test_priorities_are_ordered_consistently():
    from fewshot_vi.guidance import engine as g
    assert g.PRIORITY_SYSTEM > g.PRIORITY_ANSWER > g.PRIORITY_FIND > g.PRIORITY_ANNOUNCE > g.PRIORITY_HINT > g.PRIORITY_BEEP


# ----------------------------------------------------------------------- TTS
class GateBackend(tts.TTSBackend):
    """Blocks each utterance until released; records order and stop() calls."""
    def __init__(self, block=True):
        self.spoken, self.stops, self.block = [], 0, block
        self.release = threading.Event()
        self.started = threading.Event()
        if not block:
            self.release.set()

    def speak(self, text):
        self.started.set()
        self.release.wait(2)
        self.spoken.append(text)

    def stop(self):
        self.stops += 1
        self.release.set()


def test_say_never_blocks_the_caller():
    be = GateBackend()
    sp = tts.Speaker(be)
    t0 = time.perf_counter()
    for i in range(20):
        sp.say(f"m{i}", priority=10)
    assert time.perf_counter() - t0 < 0.1
    be.release.set()
    sp.close()


def test_highest_priority_is_spoken_first_after_current():
    be = GateBackend()
    sp = tts.Speaker(be, stale_s=100)
    sp.say("first", 10)
    assert be.started.wait(1)
    sp.say("hint", 5)
    sp.say("answer", 30)
    sp.say("announce", 10)
    be.release.set()
    assert sp.wait_idle(2)
    assert be.spoken[0] == "first" and be.spoken[1:] == ["answer", "announce", "hint"]
    sp.close()


def test_higher_priority_interrupts_lower_priority_speech():
    be = GateBackend()
    sp = tts.Speaker(be)
    sp.say("hint text", 5)
    assert be.started.wait(1)
    sp.say("urgent answer", 30)
    assert sp.wait_idle(2)
    assert be.stops == 1 and sp.interrupted == 1
    assert "urgent answer" in be.spoken
    sp.close()


def test_equal_or_lower_priority_does_not_interrupt():
    be = GateBackend()
    sp = tts.Speaker(be)
    sp.say("announce", 10)
    assert be.started.wait(1)
    sp.say("another announce", 10)
    sp.say("hint", 5)
    assert be.stops == 0
    be.release.set()
    sp.close()


def test_stale_low_priority_speech_is_dropped_but_answers_are_kept():
    clk = FakeClock(0.0)
    be = GateBackend()
    sp = tts.Speaker(be, stale_s=2.0, stale_below_priority=20, clock=clk)
    sp.say("blocker", 10)
    assert be.started.wait(1)
    sp.say("old hint", 5)
    sp.say("old announce", 10)
    sp.say("old answer", 30)
    clk.advance(3.0)
    be.release.set()
    assert sp.wait_idle(2)
    assert "old hint" not in be.spoken and "old announce" not in be.spoken
    assert "old answer" in be.spoken and sp.dropped_stale == 2
    sp.close()


def test_backend_failure_falls_back_and_worker_survives(capsys):
    class Boom(tts.TTSBackend):
        def speak(self, text):
            raise RuntimeError("engine died")
    sp = tts.Speaker(Boom())
    sp.say("hello", 10)
    assert sp.wait_idle(2)
    assert sp.errors == 1 and "[speech] hello" in capsys.readouterr().out
    assert sp.spoken == []  # a failed utterance is NOT recorded as spoken
    sp.say("again", 10)
    assert sp.wait_idle(2) and sp.errors == 2
    sp.close()


def test_empty_text_rejected_and_queue_bounded():
    be = GateBackend()
    sp = tts.Speaker(be, max_queue=3)
    assert sp.say("   ") is False and sp.say("") is False
    sp.say("a", 10)
    assert be.started.wait(1)
    for i in range(10):
        sp.say(f"low{i}", 5)
    assert sp.pending() <= 3
    assert sp.say("vip", 30) is True
    be.release.set()
    sp.close()


def test_console_fallback_prints(capsys):
    tts.ConsoleBackend().speak("hi")
    assert "[speech] hi" in capsys.readouterr().out


def test_make_backend_falls_back_to_console_when_engine_missing(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pyttsx3":
            raise ImportError("no pyttsx3")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert tts.make_backend().name == "console"


# --------------------------------------------------------------------- beeps
def dominant_hz(sig, sr):
    mono = sig.sum(axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)), n=len(mono) * 8))
    return np.argmax(spec) * sr / (len(mono) * 8)


def test_beep_is_stereo_panned_and_pitch_rises():
    sr = 22050
    left, right, mid = beeps.make_beep(-1, 0.0, sr), beeps.make_beep(1, 0.0, sr), beeps.make_beep(0, 0.0, sr)
    assert left.shape[1] == 2 and left.dtype == np.float32
    assert np.abs(left[:, 0]).max() > 10 * np.abs(left[:, 1]).max()
    assert np.abs(right[:, 1]).max() > 10 * np.abs(right[:, 0]).max()
    assert np.abs(mid[:, 0]).max() == pytest.approx(np.abs(mid[:, 1]).max(), rel=1e-3)
    assert dominant_hz(beeps.make_beep(0, 1.0, sr), sr) > dominant_hz(beeps.make_beep(0, 0.0, sr), sr) + 200
    assert np.abs(beeps.make_beep(5, 9, sr)).max() <= 0.41  # out-of-range inputs are clamped


def test_beeper_plays_and_degrades_gracefully():
    got = []
    b = beeps.Beeper(player=lambda d, sr: got.append((d.shape, sr)))
    assert b.beep(0.5, 0.5) is True and b.played == 1 and got

    def boom(d, sr):
        raise OSError("device gone")
    b2 = beeps.Beeper(player=boom)
    assert b2.beep(0, 0) is False and b2.available is False
    assert b2.beep(0, 0) is False  # stays disabled, never raises


def test_beeper_without_audio_device(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    b = beeps.Beeper()
    assert b.available is False and b.beep(0, 0) is False


# ----------------------------------------------------------------------- STT
def test_keyboard_stt_reads_lines_and_signals_eof():
    k = stt.KeyboardSTT(io.StringIO("find keys\n\nstop\n"), printer=lambda s: None)
    assert k.listen() == "find keys"
    assert k.listen() is None and not k.eof      # blank line
    assert k.listen() == "stop"
    assert k.listen() is None and k.eof


def test_keyboard_console_timeout_returns_so_listener_can_stop(monkeypatch, capsys):
    import select
    monkeypatch.setattr(stt.sys, "platform", "linux")
    monkeypatch.setattr(select, "select", lambda *args: ([], [], []))
    k = stt.KeyboardSTT(printer=lambda s: None)
    assert k.listen(timeout=0.01) is None
    assert not k.eof


def test_make_stt_falls_back_to_keyboard_when_microphone_unusable(monkeypatch):
    class BadMic:
        def __init__(self, *a, **k):
            raise RuntimeError("microphone STT unavailable: no device")
    monkeypatch.setattr(stt, "MicrophoneSTT", BadMic)
    assert stt.make_stt(True).name == "keyboard"
    assert stt.make_stt(False).name == "keyboard"


def test_voice_listener_runs_off_thread_and_delivers_text():
    got, main = [], threading.current_thread()
    seen_threads = []
    k = stt.KeyboardSTT(io.StringIO("find keys\nwhat is this\n"), printer=lambda s: None)

    def on_text(t):
        got.append(t)
        seen_threads.append(threading.current_thread())
    vl = stt.VoiceListener(k, on_text).start()
    end = time.time() + 2
    while len(got) < 2 and time.time() < end:
        time.sleep(0.01)
    vl.stop()
    assert got == ["find keys", "what is this"] and all(t is not main for t in seen_threads)


def test_voice_listener_survives_handler_exceptions():
    k = stt.KeyboardSTT(io.StringIO("a\nb\n"), printer=lambda s: None)
    got = []

    def on_text(t):
        got.append(t)
        if t == "a":
            raise ValueError("bad handler")
    vl = stt.VoiceListener(k, on_text).start()
    end = time.time() + 2
    while len(got) < 2 and time.time() < end:
        time.sleep(0.01)
    vl.stop()
    assert got == ["a", "b"]


def test_reply_queue():
    r = stt.ReplyQueue()
    assert r.get(0.01) is None
    r.put("x")
    assert r.get(0.1) == "x"
    r.put("y")
    r.drain()
    assert r.get(0.01) is None


# --------------------------------------------------------------------- hands
@pytest.mark.parametrize("hand,expected", [
    ((300, 200), "Stop, it's right there."),
    ((100, 200), "Move your hand right."),
    ((250, 200), "Move your hand slightly right."),
    ((500, 200), "Move your hand left."),
    ((300, 40), "Move your hand down."),
    ((100, 40), "Move your hand right and down."),
])
def test_reach_instruction(hand, expected):
    assert reach_instruction(hand, (280, 180, 320, 220), near_px=40) == expected
