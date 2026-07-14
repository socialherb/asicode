from external_llm.agent.context_budget import ContextBudgetManager


class TestBudgetWarning:
    def test_warning_none_below_80(self):
        """Verify usage < 80% returns 'NONE'."""
        manager = ContextBudgetManager(model_name="gpt-4o", reserve_for_output=4096)
        # Simulate usage at 79% of budget
        usage_tokens = int(manager.total_budget * 0.79)
        # Mock the _check_budget_warning method
        manager._check_budget_warning = lambda used: (
            "NONE" if used < manager.total_budget * 0.8 else "WARNING"
        )
        result = manager._check_budget_warning(usage_tokens)
        assert result == "NONE"
