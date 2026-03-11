# Project Instructions

## Workflow
- **Test first, then change**: Before making code changes, write and run tests to confirm the current behavior and validate assumptions. Only then proceed with implementation.
- When adding new functionality that depends on external services (InfluxDB, MQTT, APIs), create a standalone test script first to verify connectivity and data availability before integrating into the pipeline.

## Project Context
- Fly experiment enclosure controlled by ESPHome (ESP32) with BME680 sensors, LED light cycle, fan, heater
- Two Pi recording rigs: Pi 1 (`combinedv2_1_pi1.py`) and Pi 2 (`combinedv2_1.py`)
- Home Assistant at `10.229.137.171:8123` with InfluxDB at `10.229.137.171:8086` (user: `homeassistant`)
- ESPHome config: `esphomeflynursery2.yaml`

## InfluxDB
- Database: `homeassistant`, auth required (credentials in `.env` file, never commit secrets)
- Entity IDs: `flynursery2_temperature_fly_2`, `flynursery2_temperature_room`, `flynursery2_humidity_fly_2`, `flynursery2_humidity_room`, `flynursery2_pressure_fly_2`, `flynursery2_pressure_room`, `flynursery2_fly_sun_brightness`, `flynursery2_heat_pad_watts`, `flynursery2_average_rpm_2`
- HA InfluxDB config uses explicit entity include list in `configuration.yaml` - new sensors must be added there
- Measurements by unit: `°C`, `%`, `hPa`, `RPM`, `W`
