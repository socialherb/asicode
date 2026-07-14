"""Tests for edit_localization.dataflow_extractor."""

from external_llm.edit_localization.dataflow_extractor import (
    extract_flow_facts,
)


class TestExtractFlowFacts:
    """Test dataflow fact extraction from function bodies."""

    def test_pure_delegation(self):
        """Function that only calls another and returns result."""
        source = '''
def parse_definitions(source: str):
    tree = ast.parse(source)
    results = []
    _walk_definitions(tree, results, parent_class=None)
    results.sort(key=lambda d: d.start_line)
    return results
'''
        facts = extract_flow_facts(source)
        assert "pure_delegation" not in facts.tags  # has assignments → not pure
        assert "results" in facts.return_names
        assert "tree" in facts.assigned_names

    def test_true_pure_delegation(self):
        """Function that literally just delegates."""
        source = '''
def get_data(path):
    return _load_data(path)
'''
        facts = extract_flow_facts(source)
        assert "pure_delegation" in facts.tags
        assert "_load_data" in facts.delegation_calls
        assert "path" in facts.param_names

    def test_value_determiner(self):
        """Function that assigns string literal values."""
        source = '''
def _walk_definitions(tree, results, parent_class):
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.AsyncFunctionDef):
            kind = "async_function"
        else:
            kind = "function"
        results.append(DefinitionInfo(kind=kind, name=child.name))
'''
        facts = extract_flow_facts(source)
        assert "value_determiner" in facts.tags
        assert "async_function" in facts.string_literals
        assert "function" in facts.string_literals
        assert "kind" in facts.assigned_names

    def test_derivation_chain(self):
        """Variable derived from another variable."""
        source = '''
def process(node):
    is_async = isinstance(node, ast.AsyncFunctionDef)
    kind = "async_function" if is_async else "function"
    return DefinitionInfo(kind=kind)
'''
        facts = extract_flow_facts(source)
        assert "kind" in facts.derives_from
        assert "is_async" in facts.derives_from["kind"]
        assert "conditional_logic" in facts.tags

    def test_constructor_fields(self):
        """Constructor call with keyword arguments."""
        source = '''
def build_info(name, kind, start):
    return DefinitionInfo(name=name, kind=kind, start_line=start)
'''
        facts = extract_flow_facts(source)
        assert "DefinitionInfo" in facts.constructor_calls
        fields = facts.constructor_calls["DefinitionInfo"]
        assert "name" in fields
        assert "kind" in fields
        assert "start_line" in fields
        assert "field_constructor" in facts.tags

    def test_attribute_writes(self):
        """Attribute assignment on self or objects."""
        source = '''
def configure(self, mode, level):
    self.mode = mode
    self.level = level
    self.active = True
'''
        facts = extract_flow_facts(source)
        assert "mode" in facts.attribute_writes
        assert "level" in facts.attribute_writes
        assert "active" in facts.attribute_writes

    def test_string_comparisons(self):
        """String constants in comparisons."""
        source = '''
def check_type(node):
    if node.type == "class":
        return True
    elif node.type == "function":
        return False
'''
        facts = extract_flow_facts(source)
        assert "class" in facts.string_literals or "function" in facts.string_literals

    def test_ternary_string_literals(self):
        """String literals in ternary expressions."""
        source = '''
def get_kind(is_async):
    return "async_function" if is_async else "function"
'''
        facts = extract_flow_facts(source)
        assert "async_function" in facts.string_literals
        assert "function" in facts.string_literals

    def test_pass_through(self):
        """Function that returns parameter with no transformation."""
        source = '''
def identity(data):
    return data
'''
        facts = extract_flow_facts(source)
        assert "pass_through" in facts.tags
        assert "data" in facts.param_names
        assert "data" in facts.return_names

    def test_empty_function(self):
        """Empty/trivial function."""
        source = '''
def noop():
    pass
'''
        facts = extract_flow_facts(source)
        assert not facts.string_literals
        assert not facts.assigned_names
        assert not facts.derives_from

    def test_syntax_error_returns_empty(self):
        """Invalid Python returns empty facts."""
        facts = extract_flow_facts("def broken(")
        assert not facts.tags
        assert not facts.string_literals

    def test_complex_derivation(self):
        """Multi-step derivation chain."""
        source = '''
def transform(raw_input):
    cleaned = raw_input.strip()
    normalized = cleaned.lower()
    category = "type_a" if normalized.startswith("a") else "type_b"
    return Result(category=category)
'''
        facts = extract_flow_facts(source)
        assert "category" in facts.derives_from
        assert "normalized" in facts.derives_from["category"]
        assert "type_a" in facts.string_literals
        assert "type_b" in facts.string_literals
        assert "Result" in facts.constructor_calls

    def test_collection_builder(self):
        """Function that builds a collection with constructed items."""
        source = '''
def collect_items(nodes):
    items = []
    for node in nodes:
        kind = "class" if isinstance(node, ClassDef) else "function"
        items.append(Item(kind=kind, name=node.name))
    return items
'''
        facts = extract_flow_facts(source)
        assert "value_determiner" in facts.tags
        assert "class" in facts.string_literals
        assert "function" in facts.string_literals
        assert "Item" in facts.constructor_calls

    def test_no_noise_tokens(self):
        """Noise tokens like 'self', 'cls' are excluded."""
        source = '''
def method(self, cls):
    self.value = "important"
    return None
'''
        facts = extract_flow_facts(source)
        assert "self" not in facts.param_names
        assert "cls" not in facts.param_names
        assert "important" in facts.string_literals


class TestAliasChains:
    """Test alias_chains field — heap-level object identity."""

    def test_direct_alias_captured(self):
        """u2 = u1 should record alias_chains['u2'] = 'u1'."""
        source = '''
def process(user):
    u1 = get_user(1)
    u2 = u1
    return u2
'''
        facts = extract_flow_facts(source)
        assert "u2" in facts.alias_chains
        assert facts.alias_chains["u2"] == "u1"

    def test_multi_hop_alias(self):
        """u3 = u2; u2 = u1 → chain: u3 → u2 → u1."""
        source = '''
def chain():
    u1 = get_user(1)
    u2 = u1
    u3 = u2
    return u3
'''
        facts = extract_flow_facts(source)
        assert facts.alias_chains.get("u2") == "u1"
        assert facts.alias_chains.get("u3") == "u2"

    def test_computed_not_alias(self):
        """x = y + z is NOT a single-hop alias (computed value)."""
        source = '''
def compute(a, b):
    result = a + b
    return result
'''
        facts = extract_flow_facts(source)
        assert "result" not in facts.alias_chains
        # derives_from handles this
        assert "result" in facts.derives_from

    def test_call_result_not_alias(self):
        """u1 = get_user(1) is not in alias_chains (it's a call result)."""
        source = '''
def fetch():
    u1 = get_user(1)
    return u1
'''
        facts = extract_flow_facts(source)
        # u1 comes from a call, not a Name → not an alias
        assert "u1" not in facts.alias_chains

    def test_noise_names_excluded(self):
        """Assignments involving self/cls/None are excluded from alias chains."""
        source = '''
def configure(self):
    obj = self
    return obj
'''
        facts = extract_flow_facts(source)
        # self is a noise name — excluded
        assert "obj" not in facts.alias_chains


class TestResolveAliasRoot:
    """Test resolve_alias_root helper."""

    def test_direct_chain(self):
        from external_llm.edit_localization.dataflow_extractor import resolve_alias_root
        chains = {"u2": "u1"}
        assert resolve_alias_root("u2", chains) == "u1"

    def test_multi_hop(self):
        from external_llm.edit_localization.dataflow_extractor import resolve_alias_root
        chains = {"u3": "u2", "u2": "u1"}
        assert resolve_alias_root("u3", chains) == "u1"

    def test_no_alias(self):
        from external_llm.edit_localization.dataflow_extractor import resolve_alias_root
        chains = {}
        assert resolve_alias_root("u1", chains) == "u1"

    def test_cycle_safe(self):
        from external_llm.edit_localization.dataflow_extractor import resolve_alias_root
        chains = {"a": "b", "b": "a"}  # cycle
        result = resolve_alias_root("a", chains)
        # Should not hang; returns whatever it reaches
        assert result in {"a", "b"}


class TestCallSites:
    """Test call_sites field — object identity extraction."""

    def test_literal_args_captured(self):
        """get_user(1) and get_user(2) should produce two call-site entries."""
        source = '''
def build(repo):
    u1 = get_user(1)
    u2 = get_user(2)
    return u1, u2
'''
        facts = extract_flow_facts(source)
        assert "get_user" in facts.call_sites
        sites = facts.call_sites["get_user"]
        # Two calls, each with one literal arg
        assert len(sites) == 2
        arg_values = [s[0] for s in sites if s]
        assert "1" in arg_values
        assert "2" in arg_values

    def test_string_literal_args_captured(self):
        """fetch('admin') should record 'admin' as call-site arg."""
        source = '''
def setup():
    return fetch("admin")
'''
        facts = extract_flow_facts(source)
        assert "fetch" in facts.call_sites
        sites = facts.call_sites["fetch"]
        assert len(sites) == 1
        assert "'admin'" in sites[0]

    def test_expression_args_not_captured(self):
        """Dynamic args like get_user(x) produce empty arg list (not skipped)."""
        source = '''
def wrap(x):
    return get_user(x)
'''
        facts = extract_flow_facts(source)
        assert "get_user" in facts.call_sites
        sites = facts.call_sites["get_user"]
        # Call recorded but args=[] (expression, not literal)
        assert sites == [[]]

    def test_no_args_call_recorded(self):
        """flush() with no args still records an empty arg entry."""
        source = '''
def commit():
    db.flush()
    db.commit()
'''
        facts = extract_flow_facts(source)
        # Method calls resolved to attr name
        assert "flush" in facts.call_sites or "commit" in facts.call_sites

    def test_multiple_callees_tracked(self):
        """Different callees tracked independently."""
        source = '''
def process():
    a = load("x")
    b = load("y")
    c = save(42)
'''
        facts = extract_flow_facts(source)
        assert "load" in facts.call_sites
        assert len(facts.call_sites["load"]) == 2
        assert "save" in facts.call_sites
        assert facts.call_sites["save"] == [["42"]]
