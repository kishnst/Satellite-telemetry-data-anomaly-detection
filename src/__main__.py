from .telemetry_simulator import generate_telemetry

if __name__ == "__main__":
    data = generate_telemetry()
    print(data.head())
    print(f"Anomaly rate: {data['is_anomaly'].mean():.2%}")
