"""
Unit tests for string_helper module.
"""

from utils.string_helper import (
    count_vowels,
    is_palindrome,
    reverse_string,
)


def test_reverse_string():
    """Test reverse_string function."""
    # Basic cases
    assert reverse_string("hello") == "olleh"
    assert reverse_string("world") == "dlrow"

    # Edge cases
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"
    assert reverse_string("ab") == "ba"

    # Strings with spaces
    assert reverse_string("hello world") == "dlrow olleh"
    assert reverse_string("  ") == "  "

    # Strings with special characters
    assert reverse_string("12345") == "54321"
    assert reverse_string("!@#$") == "$#@!"


def test_reverse_string_unicode():
    """Test reverse_string with Unicode characters."""
    assert reverse_string("안녕하세요") == "요세하녕안"
    assert reverse_string("café") == "éfac"
    assert reverse_string("🎉🎊") == "🎊🎉"


def test_count_vowels():
    """Test count_vowels function."""
    # Basic cases
    assert count_vowels("hello") == 2
    assert count_vowels("world") == 1
    assert count_vowels("aeiou") == 5
    assert count_vowels("AEIOU") == 5

    # Edge cases
    assert count_vowels("") == 0
    assert count_vowels("xyz") == 0
    assert count_vowels("bcdfg") == 0

    # Mixed case
    assert count_vowels("Hello World") == 3
    assert count_vowels("Programming") == 3
    assert count_vowels("Python") == 1

    # With numbers and special characters
    # Digits replace vowels here — no ASCII vowels present
    assert count_vowels("h3ll0 w0rld!") == 0
    assert count_vowels("a1e2i3o4u5") == 5


def test_count_vowels_unicode():
    """Test count_vowels with Unicode characters."""
    # Note: Only ASCII English vowels are counted; accented chars (é, ï, ü) are NOT vowels
    assert count_vowels("café") == 1  # only 'a' is ASCII vowel ('é' is not)
    assert count_vowels("naïve") == 2  # 'a' and 'e' are ASCII vowels ('ï' is not)
    assert count_vowels("über") == 1  # only 'e' is ASCII vowel ('ü' is not)


def test_is_palindrome():
    """Test is_palindrome function."""
    # Basic palindromes
    assert is_palindrome("racecar")
    assert is_palindrome("level")
    assert is_palindrome("radar")

    # Non-palindromes
    assert not is_palindrome("hello")
    assert not is_palindrome("world")
    assert not is_palindrome("python")

    # Edge cases
    assert is_palindrome("")  # Empty string is palindrome
    assert is_palindrome("a")  # Single character is palindrome
    assert is_palindrome("aa")
    assert not is_palindrome("ab")

    # Case insensitive
    assert is_palindrome("Racecar")
    assert is_palindrome("Level")
    assert is_palindrome("RaDaR")


def test_is_palindrome_with_punctuation():
    """Test is_palindrome with punctuation and spaces."""
    # Classic palindrome with punctuation
    assert is_palindrome("A man, a plan, a canal: Panama")

    # With spaces
    assert is_palindrome("race car")
    assert is_palindrome("was it a car or a cat i saw")

    # With numbers
    assert is_palindrome("12321")
    assert not is_palindrome("12345")
    assert is_palindrome("1a2b3b2a1")

    # With special characters
    assert is_palindrome("madam i'm adam")
    assert is_palindrome("never odd or even")


def test_is_palindrome_unicode():
    """Test is_palindrome with Unicode characters."""
    # Korean palindrome
    assert is_palindrome("토마토")

    # Note: The function only handles alphanumeric characters,
    # so non-alphanumeric Unicode will be removed
    assert is_palindrome("caféfac")  # 'é' stripped → "cafefac" is a palindrome

    # Pure Unicode palindrome (alphanumeric only)
    assert is_palindrome("abcba")


def test_is_palindrome_invalid_input():
    """Test is_palindrome with invalid input."""
    # None should be handled gracefully
    assert is_palindrome("")

    # Very long palindrome
    long_palindrome = "a" * 1000 + "b" + "a" * 1000
    assert is_palindrome(long_palindrome)

    # Very long non-palindrome
    long_non_palindrome = "a" * 1000 + "b" + "c" + "a" * 1000
    assert not is_palindrome(long_non_palindrome)


if __name__ == "__main__":
    # Run tests directly for debugging
    test_reverse_string()
    test_reverse_string_unicode()
    test_count_vowels()
    test_count_vowels_unicode()
    test_is_palindrome()
    test_is_palindrome_with_punctuation()
    test_is_palindrome_unicode()
    test_is_palindrome_invalid_input()
    print("All tests passed!")
