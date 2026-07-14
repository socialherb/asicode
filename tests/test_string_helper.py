"""
Comprehensive pytest unit tests for string_helper module.

Tests the three functions in utils/string_helper.py with various inputs
including edge cases (empty string, strings with special characters, mixed case).
"""
import pytest

from utils.string_helper import (
    count_vowels,
    is_palindrome,
    reverse_string,
)


class TestReverseString:
    """Test suite for reverse_string function."""

    def test_basic_reversal(self):
        """Test basic string reversal."""
        assert reverse_string("hello") == "olleh"
        assert reverse_string("world") == "dlrow"
        assert reverse_string("python") == "nohtyp"

    def test_edge_cases(self):
        """Test edge cases for reverse_string."""
        # Empty string
        assert reverse_string("") == ""

        # Single character
        assert reverse_string("a") == "a"
        assert reverse_string("Z") == "Z"

        # Two characters
        assert reverse_string("ab") == "ba"
        assert reverse_string("12") == "21"

    def test_strings_with_spaces(self):
        """Test strings containing spaces."""
        assert reverse_string("hello world") == "dlrow olleh"
        assert reverse_string("  ") == "  "
        assert reverse_string("a b c") == "c b a"

    def test_strings_with_special_characters(self):
        """Test strings with special characters."""
        assert reverse_string("12345") == "54321"
        assert reverse_string("!@#$") == "$#@!"
        assert reverse_string("a@b#c") == "c#b@a"

    def test_mixed_case(self):
        """Test strings with mixed case."""
        assert reverse_string("Hello") == "olleH"
        assert reverse_string("PyThOn") == "nOhTyP"
        assert reverse_string("AbCdEf") == "fEdCbA"

    def test_unicode_characters(self):
        """Test reverse_string with Unicode characters."""
        assert reverse_string("안녕하세요") == "요세하녕안"
        assert reverse_string("café") == "éfac"
        assert reverse_string("🎉🎊") == "🎊🎉"


class TestCountVowels:
    """Test suite for count_vowels function."""

    def test_basic_vowel_count(self):
        """Test basic vowel counting."""
        assert count_vowels("hello") == 2
        assert count_vowels("world") == 1
        assert count_vowels("aeiou") == 5

    def test_case_insensitivity(self):
        """Test that vowel counting is case insensitive."""
        assert count_vowels("AEIOU") == 5
        assert count_vowels("Hello World") == 3
        assert count_vowels("PyThOn") == 1
        assert count_vowels("AaEeIiOoUu") == 10

    def test_edge_cases(self):
        """Test edge cases for count_vowels."""
        # Empty string
        assert count_vowels("") == 0

        # No vowels
        assert count_vowels("xyz") == 0
        assert count_vowels("bcdfg") == 0
        assert count_vowels("12345") == 0
        assert count_vowels("!@#$") == 0

    def test_strings_with_non_vowel_characters(self):
        """Test strings with mixed vowel and non-vowel characters."""
        assert count_vowels("h3ll0 w0rld!") == 0  # digits are not vowels
        assert count_vowels("a1e2i3o4u5") == 5
        assert count_vowels("programming") == 3
        assert count_vowels("beautiful") == 5
        assert count_vowels("a1e2i3o4u5") == 5
        assert count_vowels("programming") == 3
        assert count_vowels("beautiful") == 5

    def test_unicode_characters(self):
        """Test count_vowels with Unicode characters."""
        assert count_vowels("café") == 1  # 'a' is vowel; 'é' is non-ASCII
        assert count_vowels("naïve") == 2  # 'a' and 'i' are vowels; 'ë' stripped
        assert count_vowels("über") == 1  # 'e' is vowel; 'ü' stripped


class TestIsPalindrome:
    """Test suite for is_palindrome function."""

    def test_true_palindromes(self):
        """Test strings that are palindromes."""
        assert is_palindrome("racecar")
        assert is_palindrome("level")
        assert is_palindrome("radar")
        assert is_palindrome("madam")

    def test_false_cases(self):
        """Test strings that are not palindromes."""
        assert not is_palindrome("hello")
        assert not is_palindrome("world")
        assert not is_palindrome("python")
        assert not is_palindrome("programming")

    def test_edge_cases(self):
        """Test edge cases for is_palindrome."""
        # Empty string is palindrome
        assert is_palindrome("")

        # Single character is palindrome
        assert is_palindrome("a")
        assert is_palindrome("Z")

        # Two characters
        assert is_palindrome("aa")
        assert not is_palindrome("ab")

    def test_case_insensitivity(self):
        """Test that palindrome checking ignores case."""
        assert is_palindrome("Racecar")
        assert is_palindrome("Level")
        assert is_palindrome("RaDaR")
        assert is_palindrome("MadAm")

    def test_ignoring_non_alphanumeric_characters(self):
        """Test that palindrome checking ignores non-alphanumeric characters."""
        # Classic palindrome with punctuation
        assert is_palindrome("A man, a plan, a canal: Panama")

        # With spaces
        assert is_palindrome("race car")
        assert is_palindrome("was it a car or a cat i saw")

        # With special characters
        assert is_palindrome("madam i'm adam")
        assert is_palindrome("never odd or even")

    def test_strings_with_numbers(self):
        """Test palindrome checking with numbers."""
        assert is_palindrome("12321")
        assert not is_palindrome("12345")
        assert is_palindrome("1a2b3b2a1")
        assert is_palindrome("1234321")

    def test_unicode_palindromes(self):
        """Test palindrome checking with Unicode characters."""
        # Korean palindrome
        assert is_palindrome("토마토")

        # Note: The function only handles alphanumeric characters,
        # so non-alphanumeric Unicode will be removed
        # 'é' stripped -> 'cafac' IS a palindrome
        assert is_palindrome("caféfac")  # 'é' will be removed

    def test_long_strings(self):
        """Test palindrome checking with long strings."""
        # Very long palindrome
        long_palindrome = "a" * 1000 + "b" + "a" * 1000
        assert is_palindrome(long_palindrome)

        # Very long non-palindrome
        long_non_palindrome = "a" * 1000 + "b" + "c" + "a" * 1000
        assert not is_palindrome(long_non_palindrome)


def test_all_functions_integration():
    """Integration test using all three functions together."""
    test_string = "A man, a plan, a canal: Panama"

    # This is a palindrome
    assert is_palindrome(test_string)

    # Count vowels in the string
    assert count_vowels(test_string) == 10  # 10 'a's in the palindrome  # a, a, a, a, a, a, a, a, a, a, a, a

    # Reverse the string
    reversed_str = reverse_string(test_string)
    assert reversed_str == "amanaP :lanac a ,nalp a ,nam A"

    # The reversed string IS a palindrome (cleaned: amanaplanacanalpanama)
    assert is_palindrome(reversed_str)


if __name__ == "__main__":
    # Run tests directly for debugging
    pytest.main([__file__, "-v"])
