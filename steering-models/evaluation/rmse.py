

def calc_rmse(prediction_fn, data_iter, *args):
    mse = 0.
    count = 0
    for image_pred, image_disp, speed, steering, ts in data_iter:
        count += 1
        predicted_steering = prediction_fn(image_disp)
        mse += (steering - predicted_steering)**2.
        if count % 50 == 0:
            print count, ':', (mse/count)**0.5
    return (mse/count) ** 0.5
