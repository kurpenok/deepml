from solutions.s0092_linear_regression_power_grid_optimization import (
    power_grid_forecast,
)


def test_case_1():
    assert (
        power_grid_forecast([150, 165, 185, 195, 210, 225, 240, 260, 275, 290]) == 404
    )


def test_case_2():
    assert (
        power_grid_forecast([160, 170, 190, 200, 215, 230, 245, 265, 280, 295]) == 407
    )


def test_case_3():
    assert (
        power_grid_forecast([140, 158, 180, 193, 205, 220, 237, 255, 270, 288]) == 404
    )
