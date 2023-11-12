
import numpy as np
from dataframes import create_array
import time

departure_time_train, stops_train, arrival_time_train, class_train, duration_train, days_left_train, price_train, \
departure_time_test, stops_test, arrival_time_test, class_test, duration_test, days_left_test, \
price_test = create_array()


def mse(y_train, y_predicted):
    mse_error = np.sum((y_train - y_predicted) ** 2) / (2 * len(y_train))
    return mse_error


def rmse(y_train, y_predicted):
    rmse_error = np.sqrt(mse(y_train, y_predicted))
    return rmse_error


def mae(y_train, y_predicted):
    mae_error = np.sum(np.abs(y_train - y_predicted)) / (2 * len(y_train))
    return mae_error


def r_squared(y_train, y_predicted):
    r_squared_error = 1 - (np.sum((y_train - y_predicted) ** 2) / np.sum((y_train - np.average(y_train)) ** 2))
    return r_squared_error


def w_updating(w, n, x, y, y_predicted, learning_rate):
    w_derivative = -(2 / n) * sum(x * (y - y_predicted))
    w = w - (learning_rate * w_derivative)
    return w


def b_updating(b, n, y, y_predicted, learning_rate):
    b_derivative = -(2 / n) * sum(y - y_predicted)
    b = b - (learning_rate * b_derivative)
    return b


def price_predicted_cal(w_arrival_time, w_departure_time, w_stops, w_class_type, w_duration, w_days_left,
                        arrival_time, departure_time, stops, class_type, duration, days_left, b):
    price_predicted = (w_arrival_time * arrival_time) + (w_departure_time * departure_time) + (w_stops * stops) + (
            w_class_type * class_type) + (w_duration * duration) + (w_days_left * days_left) + b

    return price_predicted


def gradient_descent(departure_time, stops, arrival_time, class_type, duration, days_left,
                     price, iterations, learning_rate=0.1, stopping_threshold=1e-5):
    start_time = time.time()

    w_departure_time = 0.05
    w_stops = 0.2
    w_arrival_time = 0.05
    w_class_type = 0.6
    w_duration = 0.3
    w_days_left = 0.3
    bias = 0.01

    n = float(len(departure_time))

    r_squared_errors = []
    previous_r_squared_error = None

    for i in range(iterations):

        # Making predictions
        price_predicted = price_predicted_cal(w_arrival_time, w_departure_time, w_stops, w_class_type, w_duration,
                                              w_days_left,
                                              arrival_time, departure_time, stops, class_type, duration, days_left,
                                              bias)
        # Calculating the current cost
        r_squared_error = r_squared(price, price_predicted)

        # stopping_threshold we stop the gradient descent
        if previous_r_squared_error and abs(previous_r_squared_error - r_squared_error) <= stopping_threshold:
            break

        previous_r_squared_error = r_squared_error

        r_squared_errors.append(r_squared_error)

        w_departure_time = w_updating(w_departure_time, n, departure_time, price, price_predicted, learning_rate)
        w_stops = w_updating(w_stops, n, stops, price, price_predicted, learning_rate)
        w_arrival_time = w_updating(w_departure_time, n, arrival_time, price, price_predicted, learning_rate)
        w_class_type = w_updating(w_class_type, n, class_type, price, price_predicted, learning_rate)
        w_duration = w_updating(w_duration, n, duration, price, price_predicted, learning_rate)
        w_days_left = w_updating(w_days_left, n, days_left, price, price_predicted, learning_rate)
        bias = b_updating(bias, n, price, price_predicted, learning_rate)

    f = open("[7]-UIAI4021-PR1-Q2.txt", "w")
    f.write(f'PRICE = ({round(w_arrival_time, 2)} * arrival_time) + ({round(w_departure_time, 2)} * departure_time) + '
            f'({round(w_stops, 2)} * stops) + ({round(w_class_type, 2)} * class_type) + ({round(w_duration, 2)} * '
            f'duration) + 'f'({round(w_days_left, 2)} * days_left) + {round(bias, 2)}' + '\n')

    f.write(f'Training Time: {round(time.time() - start_time, 2)}s' + '\n\n')
    f.close()

    return w_departure_time, w_stops, w_arrival_time, w_class_type, w_duration, w_days_left, bias


def test(arrival_time, departure_time, stops, class_type, duration, days_left,
         w_arrival_time, w_departure_time, w_stops, w_class_type, w_duration, w_days_left, b):
    for i in range(len(stops_test)):
        price_predicted = price_predicted_cal(w_arrival_time, w_departure_time, w_stops, w_class_type,
                                              w_duration, w_days_left, arrival_time, departure_time,
                                              stops, class_type, duration, days_left, b)

    mse_error = mse(price_test, price_predicted)
    rmse_error = rmse(price_test, price_predicted)
    mae_error = mae(price_test, price_predicted)
    r_squared_error = r_squared(price_test, price_predicted)

    f = open("[7]-UIAI4021-PR1-Q2.txt", "a")
    f.write(f'Logs:' + '\n')
    f.write(f'MSE: {round(mse_error, 3)}' + '\n')
    f.write(f'RMSE: {round(rmse_error, 3)}' + '\n')
    f.write(f'MAE: {round(mae_error, 3)}' + '\n')
    f.write(f'R2: {round(r_squared_error, 3)}' + '\n')
    f.close()


def main():
    w_departure_time, w_stops, w_arrival_time, w_class_type, w_duration, w_days_left, bias = gradient_descent(
        departure_time_train, stops_train, arrival_time_train,
        class_train, duration_train, days_left_train,
        price_train, len(price_train))

    test(arrival_time_test, departure_time_test, stops_test, class_test, duration_test, days_left_test,
         w_arrival_time, w_departure_time, w_stops, w_class_type, w_duration, w_days_left, bias)


if __name__ == '__main__':
    main()
