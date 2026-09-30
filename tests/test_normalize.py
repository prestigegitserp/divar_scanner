from divar_scanner.normalize import parse_number


def test_parse_persian_numbers_and_units():
    assert parse_number("۲ میلیارد تومان") == 2_000_000_000
    assert parse_number("۱۵ میلیون") == 15_000_000
    assert parse_number("۸۵") == 85
    assert parse_number("توافقی") is None
