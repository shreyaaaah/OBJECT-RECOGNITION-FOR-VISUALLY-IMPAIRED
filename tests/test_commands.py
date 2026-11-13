import pytest

from fewshot_vi.voice.commands import Intent, parse_command, resolve_name

NAMES = ["keys", "wallet", "blue mug", "red mug", "medicine box"]


def p(text):
    return parse_command(text, NAMES)


@pytest.mark.parametrize("text", ["what is this", "What's this?", "identify", "please identify this",
                                  "what am I holding"])
def test_identify(text):
    assert p(text).intent == Intent.IDENTIFY


@pytest.mark.parametrize("text,name", [
    ("find my keys", "keys"), ("where is my wallet", "wallet"), ("Where's the wallet?", "wallet"),
    ("can you find my blue mug please", "blue mug"), ("look for medicine box", "medicine box"),
    ("locate keys", "keys"),
])
def test_find(text, name):
    c = p(text)
    assert c.intent == Intent.FIND and c.name == name


@pytest.mark.parametrize("text,name", [
    ("where did I leave my keys", "keys"), ("where did i put the wallet", "wallet"),
    ("when did I last see my red mug", "red mug"),
])
def test_memory_query_beats_find(text, name):
    c = p(text)
    assert c.intent == Intent.MEMORY_QUERY and c.name == name


@pytest.mark.parametrize("text", ["register object", "add object", "Register new object", "enroll",
                                  "add new object", "remember this object"])
def test_enrol(text):
    assert p(text).intent == Intent.ENROL


def test_delete_and_forget():
    c = p("delete wallet")
    assert c.intent == Intent.DELETE and c.name == "wallet"
    c = p("forget my red mug")
    assert c.intent == Intent.DELETE and c.name == "red mug"


def test_list_mute_resume():
    assert p("list objects").intent == Intent.LIST
    assert p("what objects do I have").intent == Intent.LIST
    assert p("stop").intent == Intent.MUTE and p("quiet").intent == Intent.MUTE
    assert p("resume").intent == Intent.RESUME and p("unmute").intent == Intent.RESUME


def test_confuser():
    c = p("this is similar to blue mug")
    assert c.intent == Intent.CONFUSER and c.name == "blue mug"


def test_yes_no():
    assert p("yes").intent == Intent.YES and p("yeah sure").intent == Intent.YES
    assert p("no").intent == Intent.NO and p("nope").intent == Intent.NO


@pytest.mark.parametrize("text,intent,name", [
    ("find my kees", Intent.FIND, "keys"),            # misspelled name
    ("find my walet", Intent.FIND, "wallet"),
    ("fnd my keys", Intent.FIND, "keys"),             # misspelled verb
    ("delet wallet", Intent.DELETE, "wallet"),
    ("regster object", Intent.ENROL, None),
    ("where did i leave my wallett", Intent.MEMORY_QUERY, "wallet"),
])
def test_misspellings(text, intent, name):
    c = p(text)
    assert c.intent == intent and c.name == name


def test_unresolved_name_keeps_raw_text():
    c = p("find my passport")
    assert c.intent == Intent.FIND and c.name is None and c.raw_name == "passport"
    assert c.needs_name


def test_gibberish_and_empty_are_unknown():
    assert p("banana smoothie recipe").intent == Intent.UNKNOWN
    assert p("").intent == Intent.UNKNOWN and p("   ").intent == Intent.UNKNOWN


def test_resolve_name_cases():
    assert resolve_name("KEYS", NAMES) == "keys"
    assert resolve_name("the blue mug please", NAMES) == "blue mug"
    assert resolve_name("mug", ["mug"]) == "mug"
    assert resolve_name("zzzz", NAMES) is None
    assert resolve_name("keys", []) is None
