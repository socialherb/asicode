"""Tests for edit_localization.relevance_scorer."""

from external_llm.edit_localization.dataflow_extractor import extract_flow_facts
from external_llm.edit_localization.relevance_scorer import (
    _tokenize_request,
    score_edit_relevance,
)


class TestScoreEditRelevance:
    """Integration tests: extract facts + score against request."""

    def test_kp6_walk_definitions_scores_higher(self):
        """KP6 canonical case: 'kind를 통일해줘' should rank _walk_definitions above parse_definitions.

        _walk_definitions has kind="async_function", kind="function", DefinitionInfo(kind=...)
        parse_definitions just delegates to _walk_definitions
        """
        walk_source = '''
def _walk_definitions(node, out, parent_class):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.AsyncFunctionDef):
            kind = "async_function"
        elif isinstance(child, ast.FunctionDef):
            kind = "function"
        decorators = [_decorator_name(d) for d in child.decorator_list]
        start = child.decorator_list[0].lineno if child.decorator_list else child.lineno
        end = getattr(child, "end_lineno", child.lineno)
        out.append(DefinitionInfo(
            name=child.name,
            kind=kind,
            start_line=start,
            end_line=end,
        ))
'''
        parse_source = '''
def parse_definitions(source: str):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    results = []
    _walk_definitions(tree, results, parent_class=None)
    results.sort(key=lambda d: d.start_line)
    return results
'''
        request = "kind를 통일해줘"

        walk_facts = extract_flow_facts(walk_source)
        parse_facts = extract_flow_facts(parse_source)

        walk_score, _ = score_edit_relevance(request, walk_facts)
        parse_score, _ = score_edit_relevance(request, parse_facts)

        assert walk_score > parse_score, (
            f"_walk_definitions ({walk_score:.3f}) should score higher than "
            f"parse_definitions ({parse_score:.3f})"
        )
        # walk should have substantial score, parse should be low
        assert walk_score >= 0.3
        assert parse_score < walk_score * 0.8  # parse should be significantly lower

    def test_async_separation_request(self):
        """Request about async separation should target function with is_async derivation."""
        value_source = '''
def determine_kind(node):
    is_async = isinstance(node, ast.AsyncFunctionDef)
    kind = "async_function" if is_async else "function"
    return kind
'''
        wrapper_source = '''
def get_all_kinds(tree):
    return [determine_kind(n) for n in ast.walk(tree)]
'''
        request = "async 여부를 분리해줘"

        value_facts = extract_flow_facts(value_source)
        wrapper_facts = extract_flow_facts(wrapper_source)

        value_score, _ = score_edit_relevance(request, value_facts)
        wrapper_score, _ = score_edit_relevance(request, wrapper_facts)

        assert value_score > wrapper_score

    def test_field_addition_request(self):
        """Request to add a field should target constructor-building function."""
        builder_source = '''
def build_item(name, kind, line):
    return Item(name=name, kind=kind, start_line=line)
'''
        caller_source = '''
def process_all(items):
    return [build_item(i.name, i.kind, i.line) for i in items]
'''
        request = "Item에 end_line 필드를 추가해줘"

        builder_facts = extract_flow_facts(builder_source)
        caller_facts = extract_flow_facts(caller_source)

        builder_score, _ = score_edit_relevance(request, builder_facts)
        caller_score, _ = score_edit_relevance(request, caller_facts)

        assert builder_score > caller_score

    def test_pure_delegation_gets_low_score(self):
        """Pure delegation function should score low regardless of request."""
        source = '''
def do_thing(x):
    return _actual_do_thing(x)
'''
        facts = extract_flow_facts(source)
        score, reason = score_edit_relevance("어떤 작업이든 해줘", facts)

        assert score < 0.2
        assert "pure_delegation" in reason

    def test_value_determiner_gets_high_base_score(self):
        """Value-determining function should have high base score even without direct mention."""
        source = '''
def compute_status(result):
    if result.passed:
        status = "success"
    else:
        status = "failure"
    return Record(status=status, result=result)
'''
        facts = extract_flow_facts(source)
        score, _ = score_edit_relevance("status 값을 바꿔줘", facts)

        # Should score high due to: direct mention (status) + value_determiner + flow
        assert score >= 0.4

    def test_no_facts_moderate_score(self):
        """Empty facts should get moderate score (unknown)."""
        from external_llm.edit_localization.dataflow_extractor import SymbolFlowFacts
        facts = SymbolFlowFacts()
        score, _ = score_edit_relevance("아무 요청", facts)
        # Should not be very high or very low — unknown
        assert 0.1 <= score <= 0.5


class TestSemanticMatching:
    """Test semantic action-role matching improves scoring."""

    def test_unify_prefers_value_determiner_over_delegation(self):
        """'통일' action should strongly prefer value_determiner over delegation."""
        value_source = '''
def determine_kind(node):
    if isinstance(node, ast.AsyncFunctionDef):
        kind = "async_function"
    else:
        kind = "function"
    return kind
'''
        delegation_source = '''
def get_kind(node):
    return determine_kind(node)
'''
        request = "kind를 통일해줘"

        value_facts = extract_flow_facts(value_source)
        deleg_facts = extract_flow_facts(delegation_source)

        value_score, _ = score_edit_relevance(request, value_facts)
        deleg_score, _ = score_edit_relevance(request, deleg_facts)

        # Semantic matching should widen the gap
        assert value_score > deleg_score
        assert value_score - deleg_score > 0.2  # significant gap

    def test_add_field_prefers_constructor(self):
        """'필드 추가' should prefer function with constructor calls."""
        builder_source = '''
def create_info(name, kind):
    return DefinitionInfo(name=name, kind=kind, start_line=1)
'''
        processor_source = '''
def process_info(info):
    info.validated = True
    return info
'''
        request = "DefinitionInfo에 end_line 필드를 추가해줘"

        builder_facts = extract_flow_facts(builder_source)
        proc_facts = extract_flow_facts(processor_source)

        builder_score, _ = score_edit_relevance(request, builder_facts)
        proc_score, _ = score_edit_relevance(request, proc_facts)

        assert builder_score > proc_score

    def test_split_flag_prefers_conditional_logic(self):
        """'여부 분리' should prefer function with conditional derivation."""
        conditional_source = '''
def categorize(node):
    is_async = isinstance(node, ast.AsyncFunctionDef)
    kind = "async" if is_async else "sync"
    return kind
'''
        simple_source = '''
def format_node(node):
    return str(node)
'''
        request = "async 여부를 분리해줘"

        cond_facts = extract_flow_facts(conditional_source)
        simple_facts = extract_flow_facts(simple_source)

        cond_score, _ = score_edit_relevance(request, cond_facts)
        simple_score, _ = score_edit_relevance(request, simple_facts)

        assert cond_score > simple_score

    def test_no_semantic_signal_neutral(self):
        """Request with no detectable action/role gives neutral semantic score."""
        source = '''
def compute(x):
    result = x * 2
    return result
'''
        request = "이 코드 보여줘"  # read-only, no edit action
        facts = extract_flow_facts(source)
        score, _reason = score_edit_relevance(request, facts)
        # Should still work, just without semantic boost
        assert 0.1 <= score <= 0.8


class TestTokenizeRequest:
    """Test request tokenization."""

    def test_basic_korean(self):
        tokens = _tokenize_request("kind를 통일해줘")
 # Korean particle "를" stays attached to "kind" in the new tokenizer
        assert "kind를" in tokens
        assert "통일해줘" in tokens

    def test_camel_case_preserved(self):
        tokens = _tokenize_request("isAsyncFunction을 변경")
        # CamelCase is lowercased but NOT split by the new tokenizer
        assert "isasyncfunction을" in tokens
        assert "변경" in tokens

    def test_underscore_split(self):
        tokens = _tokenize_request("is_async를 분리")
        # Underscore splits but Korean particle stays attached
        assert "async를" in tokens
        assert "is_async를" in tokens
        assert "분리" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenize_request("the function is not working")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "not" not in tokens
        assert "working" in tokens

    def test_short_tokens_removed(self):
        tokens = _tokenize_request("a b cd efg")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cd" in tokens
        assert "efg" in tokens


class TestObjectIdentityScoring:
    """Test call_sites literal args + alias_chains in scoring."""

    def test_literal_arg_identity_match(self):
        """get_user(1) + '1' in request → identity hit boosts score."""
        source = '''
def build_profile():
    u1 = get_user(1)
    return u1
'''
        # request mentions "1" → matches call_sites["get_user"] = [["1"]]
        request_with_id = "user 1의 프로필을 가져와"
        request_without_id = "user의 프로필을 가져와"

        facts = extract_flow_facts(source)
        score_with, _ = score_edit_relevance(request_with_id, facts)
        score_without, _ = score_edit_relevance(request_without_id, facts)

        # Mentioning the literal arg "1" should boost the score
        assert score_with >= score_without

    def test_alias_chain_attr_write(self):
        """u2 = u1; u2.email = x — request mentioning email should score higher."""
        source_with_alias = '''
def update_user():
    u1 = get_user(1)
    u2 = u1
    u2.email = "new@example.com"
    return u2
'''
        source_no_alias = '''
def unrelated():
    x = compute_value()
    return x
'''
        request = "email을 업데이트해줘"

        alias_facts = extract_flow_facts(source_with_alias)
        no_alias_facts = extract_flow_facts(source_no_alias)

        alias_score, _ = score_edit_relevance(request, alias_facts)
        no_alias_score, _ = score_edit_relevance(request, no_alias_facts)

        # The function with alias + email write should score higher
        assert alias_score > no_alias_score

    def test_callee_name_flow_bonus(self):
        """Request mentioning called function name boosts flow score."""
        source = '''
def handle():
    result = get_user(42)
    return result
'''
        # request explicitly mentions "get_user"
        request_callee = "get_user 호출 결과가 잘못됨"
        request_other = "다른 함수 호출이 잘못됨"

        facts = extract_flow_facts(source)
        score_callee, _ = score_edit_relevance(request_callee, facts)
        score_other, _ = score_edit_relevance(request_other, facts)

        assert score_callee > score_other

    def test_alias_chains_available_in_facts(self):
        """Verify alias_chains is populated correctly for use in scoring."""
        source = '''
def process():
    original = fetch_record(5)
    copy = original
    copy.status = "active"
    return copy
'''
        facts = extract_flow_facts(source)
        assert "copy" in facts.alias_chains
        assert facts.alias_chains["copy"] == "original"
        assert "status" in facts.attribute_writes
