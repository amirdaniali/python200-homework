# Amir Daniali
# https://github.com/amirdaniali
# Code The Dream  PyAICC 26.4
# Week 1


"""
Pydantic models for validating weather API responses.
"""

from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Columnar hourly measurements: three parallel lists.

    Hour i is described by index i of each list, so all three lists must
    have the same length. Units come from the API's ``hourly_units``:
    temperature_2m is in degrees Celsius and precipitation is in millimetres.
    """

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_equal_lengths(self) -> "HourlyBlock":
        """Reject the block if the parallel lists differ in length.

        A length mismatch would silently pair readings with the wrong
        timestamps, producing answers that look right but are not.
        """
        n_time = len(self.time)
        n_temp = len(self.temperature_2m)
        n_precip = len(self.precipitation)
        if not (n_time == n_temp == n_precip):
            raise ValueError(
                "hourly lists must be the same length: "
                f"time={n_time}, temperature_2m={n_temp}, precipitation={n_precip}"
            )
        return self


class WeatherResponse(BaseModel):
    """Top-level Open-Meteo response for one location.

    Only the fields this pipeline needs are modeled. Other metadata in the
    payload (hourly_units, generationtime_ms, etc.) is ignored.
    """

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
