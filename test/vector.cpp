#include "lib/doctest/doctest.hpp"
#include "test_utils.hpp"
#include "vector.hpp"

TEST_SUITE ("Initialization Tests") {

    TEST_CASE ("Default initialization should make vector have all zeroes.") {

        Vector<3> actual;

        for (int i = 0; i < 3; ++i) {
            CHECK(0.0 == doctest::Approx(actual[i]));
        }
    }

    TEST_CASE ("Array initialization should make vector have the array's values.") {

        double expected[3] { 1.2, 3.4, 5.6 };
        Vector<3> actual { expected };

        for (int i = 0; i < 3; ++i) {
            CHECK(expected[i] == doctest::Approx(actual[i]));
        }
    }
}

TEST_SUITE ("Addition Tests") {

    TEST_CASE ("+= operator should add and assign.") {

        Vector<3> actual {{ 1.20, 3.40, 5.60 }};
        Vector<3> rhs {{ 7.80, 9.10, 11.12 }};
        actual += rhs;

        Vector<3> expected {{ 9.00, 12.50, 16.72 }};

        CHECK_VECTORS(expected, actual);
    }

    TEST_CASE ("+ operator should add.") {

        Vector<3> lhs {{ 10.11, 11.12, 13.14 }};
        Vector<3> rhs {{ 15.16, 17.18, 19.20 }};
        Vector<3> actual = lhs + rhs;

        Vector<3> expected {{ 25.27, 28.30, 32.34 }};

        CHECK_VECTORS(expected, actual);
    }
}

TEST_SUITE ("Subtraction Tests") {

    TEST_CASE ("-= operator should subtract and assign.") {

        Vector<3> actual {{ 1.20, 3.40, 5.60 }};
        Vector<3> rhs {{ 7.80, 9.10, 11.12 }};
        actual -= rhs;

        Vector<3> expected {{ -6.60, -5.70, -5.52 }};

        CHECK_VECTORS(expected, actual);
    }

    TEST_CASE ("- operator should subtract.") {

        Vector<3> lhs {{ 10.11, 11.12, 13.14 }};
        Vector<3> rhs {{ 15.16, 17.18, 19.20 }};
        Vector<3> actual = lhs - rhs;

        Vector<3> expected {{ -5.05, -6.06, -6.06 }};

        CHECK_VECTORS(expected, actual);
    }
}

TEST_SUITE ("Multiplication Tests") {

    double factor = 2.5;

    TEST_CASE ("*= operator should multiply and assign.") {

        Vector<3> actual {{ 1.2, 3.4, 5.6 }};
        actual *= factor;

        Vector<3> expected {{ 3, 8.5, 14 }};

        CHECK_VECTORS(actual, expected);
    }

    TEST_CASE ("* operator should multiply.") {

        Vector<3> vector {{ 7.8, 9.10, 11.12 }};
        Vector<3> actual = vector * factor;

        Vector<3> expected {{ 19.5, 22.75, 27.8 }};

        CHECK_VECTORS(actual, expected);
    }

    TEST_CASE ("Hadamard product should multiply two vectors element-wise.") {

        Vector<3> lhs {{ 1.2, 3.4, 5.6 }};
        Vector<3> rhs {{ 7.8, 9.10, 11.12 }};
        Vector<3> actual = hadamard(lhs, rhs);

        Vector<3> expected {{ 9.36, 30.94, 62.272 }};

        CHECK_VECTORS(actual, expected);
    }
}

TEST_SUITE ("Division Tests") {

    double divisor = 1.5;

    TEST_CASE ("/= operator should divide and assign.") {

        Vector<3> actual {{ 1.2, 3.4, 5.6 }};
        actual /= divisor;

        Vector<3> expected {{ 0.8, 2.26666, 3.73333 }};

        CHECK_VECTORS(actual, expected);
    }

    TEST_CASE ("/ operator should divide.") {

        Vector<3> vector {{ 7.8, 9.10, 11.12 }};
        Vector<3> actual = vector / divisor;

        Vector<3> expected {{ 5.2, 6.06666, 7.41333 }};

        CHECK_VECTORS(actual, expected);
    }
}

TEST_SUITE ("Equality Tests") {

    TEST_CASE ("== operator on same object should be true.") {

        Vector<3> vector = {{ 1.0, 2.0, 3.0 }};
        CHECK(vector == vector);
    }

    TEST_CASE ("== operator on equal vectors should be true.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.0, 2.0, 3.0 }};
        CHECK(lhs == rhs);
    }

    TEST_CASE ("== operator on nonequal vectors should be false.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.1, 2.2, 3.3 }};
        CHECK_FALSE(lhs == rhs);
    }

    TEST_CASE ("!= operator on same object should be false.") {

        Vector<3> vector = {{ 1.0, 2.0, 3.0 }};
        CHECK_FALSE(vector != vector);
    }

    TEST_CASE ("!= operator on equal vectors should be false.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.0, 2.0, 3.0 }};
        CHECK_FALSE(lhs != rhs);
    }

    TEST_CASE ("!= operator on nonequal vectors should be true.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.1, 2.2, 3.3 }};
        CHECK(lhs != rhs);
    }

    TEST_CASE ("isApprox on same object is true.") {

        Vector<3> vector = {{ 1.0, 2.0, 3.0 }};
        CHECK(isApprox(vector, vector));
    }

    TEST_CASE ("isApprox on equal vectors is true.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.0, 2.0, 3.0 }};
        CHECK(isApprox(lhs, rhs));
    }

    TEST_CASE ("isApprox on approximately equal vectors is true.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.001, 2.001, 3.001 }};
        CHECK(isApprox(lhs, rhs));
    }

    TEST_CASE ("isApprox on nonequal vectors is false.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.1, 2.1, 3.1 }};
        CHECK_FALSE(isApprox(lhs, rhs));
    }

    TEST_CASE ("isApprox on almost approximately equal vectors is false.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.01, 2.01, 3.01 }};
        CHECK_FALSE(isApprox(lhs, rhs));
    }

    TEST_CASE ("isApprox on almost approximately equal vectors is false with epsilon.") {

        Vector<3> lhs = {{ 1.0, 2.0, 3.0 }};
        Vector<3> rhs = {{ 1.00002, 2.00002, 3.00002 }};
        CHECK_FALSE(isApprox(lhs, rhs, 0.00001));
    }
}
