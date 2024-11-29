import requests

def fetch_imu_data(ip, port):
    """
    Fetch IMU data from Phyphox.
    """
    try:
        url = f"http://{ip}:{port}/get?"
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()

        # Extract acceleration and gyroscope data
        accel = data["buffer"]["acceleration"]["buffer"][:3]
        gyro = data["buffer"]["rotationrate"]["buffer"][:3]

        imu_data = {
            "accel": accel,  # Acceleration in m/s²
            "gyro": gyro     # Angular velocity in rad/s
        }
        return imu_data

    except requests.RequestException as e:
        print(f"Error fetching IMU data: {e}")
        return None
