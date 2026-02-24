from flask import Flask, request, render_template
import requests

app = Flask(__name__)

# ---------- AQI CALCULATION ----------

def calculate_aqi(pm25):
    try:
        pm25 = float(pm25)
    except Exception:
        return None

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500)
    ]

    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return round(((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo)

    return None


def get_status(aqi):
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"



@app.route("/", methods=["GET", "POST"]) 
def home():
    result = None
    if request.method == "POST":
        city = request.form.get("city", "").strip()
        if city:
            try:
                geo_resp = requests.get(
                    f"https://geocoding-api.open-meteo.com/v1/search?name={city}", timeout=5
                )
                geo = geo_resp.json()
                results = geo.get("results")
                if not results:
                    raise ValueError("No geocoding results")
                lat = results[0].get("latitude")
                lon = results[0].get("longitude")

                air_resp = requests.get(
                    f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5",
                    timeout=5,
                )
                air = air_resp.json()
                pm25 = air.get("current", {}).get("pm2_5")

                aqi = calculate_aqi(pm25)
                if aqi is None:
                    raise ValueError("PM2.5 unavailable")

                status = get_status(aqi)
                result = {"city": city, "pm25": pm25, "aqi": aqi, "status": status}
            except Exception:
                result = {"error": "City not found or API error"}
        else:
            result = {"error": "Please provide a city name"}

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run()

