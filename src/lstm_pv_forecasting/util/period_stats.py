import datetime
import numpy as np
import pandas as pd


def calculate_period_stats(dataset: pd.DataFrame,
                           stat_period_duration: datetime.timedelta,
                           latitude) -> pd.DataFrame:

    # Throw a NotImplementedError if stat_period_duration is not a day
    if stat_period_duration != datetime.timedelta(days=1):
        raise NotImplementedError(
            "stat_period_duration must currently be a day")

    # assert that the dataset has a datetime index
    assert isinstance(dataset.index, pd.DatetimeIndex)

    # Group the dataset by date and calculate the mean temperature for each day
    grouped = dataset.groupby(pd.Grouper(freq=stat_period_duration))

    result = grouped.agg(mean_temp=pd.NamedAgg(
        column='temperature', aggfunc='mean'),
        tot_irradiation=pd.NamedAgg(column='radiance', aggfunc='sum'))  # Note: radiance column is actually the hourly total irradiation in MJ/m^2

    result.loc[:, "Kt"] = result.apply(lambda x: calc_daily_clearness_index(
        x.tot_irradiation, x.name, latitude), axis=1)

    result.drop(columns=["tot_irradiation"], inplace=True)

    return result


def calc_daily_clearness_index(daily_tot_irradiation: float,
                               date: datetime.datetime,
                               latitude: float):
    """Calculate the daily clearness index.

    This function calculates the daily clearness index, which is a measure of the
    amount of solar radiation received at the Earth's surface compared to the
    amount of solar radiation that would be received under clear sky conditions.

    Args:
        daily_tot_irradiation (float): The daily total irradiation in MJ/m^2.
        date (datetime.datetime): The date for which the clearness index is calculated.
        latitude (float): The latitude of the location in degrees.

    Returns:
        float: The calculated clearness index, which is a dimensionless quantity.

    """
    Isc = 1367  # in W/mq

    # day of the year
    # TODO: what to do in leap year?
    # get day of year
    day_of_year = date.timetuple().tm_yday

    # solar declination
    y = (np.pi / 180) * ((2 * np.pi * day_of_year) / 365)
    delta = (np.pi/180)*23.45*np.sin((np.pi/180)*(360/365)*(284+day_of_year))

    # eccencicitry factor
    E0 = 1+0.033*np.cos((np.pi/180)*(360*(day_of_year))/365)

    I0n = Isc * E0

    # sunset hour angle
    omega_s = np.arccos(-np.tan(latitude*np.pi/180)*np.tan(delta))

    # Irradiation in Mj/m2
    G0 = (24*3600*1e-6/np.pi)*I0n*((np.cos(latitude*np.pi/180)*np.cos(delta)
                                    * np.sin(omega_s)+(np.sin(latitude*np.pi/180)*omega_s*np.sin(delta))))

    # clearness index
    Kt = daily_tot_irradiation / G0

    return Kt
