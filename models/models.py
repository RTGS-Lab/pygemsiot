'''
Copyright University of Minnesota 2020
Authors: Mohana Krishna, Bryan C. Runck
'''


import pandas as pd
from .utils import get_delta
from .utils import get_gamma
from .utils import get_cn
from .utils import get_cd
from .utils import solar_rad_campbell_to_metric
from .utils import solar_rad_metric_to_campbell
from .utils import get_flux_density
from .utils import get_ea
from .utils import get_es


class GDDModel:

    def __init__(self, data, timestamp_col_name, air_temp_col_name, fahrenheit):
        """
        Parameters
        ------------------------------
        data: (``pd.DataFrame``)
            The weather data with hourly readings
        timestamp_col_name: (``str``)
            The name of the timestamp column in the dataframe
        air_temp_col_name: (``str``)
            The name of the air temperature column in the dataframe
        fahrenheit: (``bool``)
            Indicates whether the temperature is in Fahrenheit units
        """

        self.timestamp_col_name = timestamp_col_name
        self.air_temp_col_name = air_temp_col_name

        # Convert the timestamp column to the pandas timestamp data type
        data.loc[:, self.timestamp_col_name] = data[self.timestamp_col_name].apply(lambda x: pd.Timestamp(x))

        # Convert the temperature to Fahrenheit if required
        if not fahrenheit:
            data.loc[:, self.air_temp_col_name] = data[self.air_temp_col_name].apply(lambda x: (x * 1.8) + 32)

        self.daily_data = pd.DataFrame(columns=['Date', 'Min_temp_F', 'Max_temp_F'])

        # Average the data over 24 hours
        grouped_data = data.groupby(data[self.timestamp_col_name].dt.date)
        dates = grouped_data.groups.keys()

        for date in dates:
            grouped_data_daily = grouped_data.get_group(date)

            min_temp = grouped_data_daily[self.air_temp_col_name].min()
            max_temp = grouped_data_daily[self.air_temp_col_name].max()

            row = [date, min_temp, max_temp]
            self.daily_data.loc[len(self.daily_data)] = row

    def growing_degree_days_barley(self, start_date_str, end_date_str):
        """
        Overview ------------------------------ This function returns a dataframe containing daily and cumulative
        growing degree days (GDD) for barley. GDD is a measurement of thermal units. Plant growth rate is correlated
        with temperature, so higher temperatures mean faster growth. Based on this concept, GDD can be mapped to crop
        development stage. [Bauer, A., Frank, A.B., and Black, A.L. (1984). Estimation of Spring Wheat Leaf Growth
        Rates and Anthesis from Air Temperature. Agron. J. 76: 829-835] For all crops, there is a base temperature
        under which plant development does not occur (GDD does not accumulate). There is also an upper limit on
        temperature. Past this point, plant development rate cannot increase any further. For barley, the Feekes
        scale is often used to describe growth stages. The Feekes scale begins at 1 (one wheat shoot visible) and
        ends at 11 (grain ripening). [Large, E.C. (1954). Growth stages in cereals - illustration of the Feekes
        scale. Plant Pathol. 3:128-129.] Haun stages are used in barley GDD calculation because high temperatures
        affect plants differently depending on the growth stage. The Haun stages, being continuous, better identify
        this threshold of when barley plant growth responds to higher temperatures. While the Feekes scale is useful,
        its discrete nature makes it difficult to discern plant growth stage during vegetative growth.
        The Haun scale consists of a single digit and a single decimal place. The first digit indicates how many
        leaves are fully developed. The decimal indicates the length of the newly developing leaf in comparison to
        the previous fully developed leaf. [Haun, J.R. (1973). Visual quantification of wheat development. Agron. J.
        65:116-119.] The barley GDD model is defined as follows: [
        https://ndawn.ndsu.nodak.edu/help-barley-growing-degree-days.html] 1. Daily Average Temperature (F) = (Daily
        Max Temp (F) + Daily Min Temp (F)) / 2 2. Daily Barley GDD = Daily Average Temp - 32F 3. Conditions: a. If
        daily Max or daily Min Temp < 32F, it's set equal to 32F b. Prior to Haun stage 2.0, (384 GDD accumulated
        since planting); If daily Max Temp > 70F then it's set equal to 70F c. After Haun stage 2.0; IF daily Max
        Temp > 95F its set equal to 95F Parameters ------------------------------ start_date_str: (``str``) The start
        date in the DD/MM/YYYY format end_date_str: (``str``) The end date in the DD/MM/YYYY format Returns
        ------------------------------ masked_data : (``pd.DataFrame``) The DataFrame containing the changed maximum
        and minimum temp values and also the GDD and accumulated GDD values
        """
        start_date = pd.Timestamp(start_date_str)
        end_date = pd.Timestamp(end_date_str)

        date_mask = (self.daily_data['Date'] >= start_date) & (self.daily_data['Date'] <= end_date)
        masked_data = self.daily_data.loc[date_mask]

        gdd_values = [0]
        agdd_values = [0]

        # If daily Max or daily Min Temp < 32 °F (0 °C) it's set equal to 32 °F (0 °C).
        masked_data['Min_temp_F'] = masked_data['Min_temp_F'].apply(lambda x: max(x, 32))
        masked_data['Max_temp_F'] = masked_data['Max_temp_F'].apply(lambda x: max(x, 32))

        cumulative_gdd = 0
        haun_stage = False

        for i in range(1, len(masked_data)):

            if haun_stage:
                masked_data['Max_temp_F'].iloc[i] = min(95, masked_data['Max_temp_F'].iloc[i])
            else:
                masked_data['Max_temp_F'].iloc[i] = min(70, masked_data['Max_temp_F'].iloc[i])

            mean_temp = (masked_data['Max_temp_F'].iloc[i] + masked_data['Min_temp_F'].iloc[i]) / 2
            gdd = mean_temp - 32

            gdd_values.append(gdd)
            cumulative_gdd += gdd

            haun_stage = cumulative_gdd >= 384
            agdd_values.append(cumulative_gdd)

        masked_data['gdd'] = gdd_values
        masked_data['agdd'] = agdd_values

        return masked_data

    def growing_degree_days_corn(self, start_date_str, end_date_str):
        """
        Overview ------------------------------ This function returns a dataframe containing daily and cumulative
        growing degree days (GDD) for corn. GDD is a measurement of thermal units. Plant growth rate is correlated
        with temperature, so higher temperatures mean faster growth. Based on this concept, GDD can be mapped to crop
        development stage. [Bauer, A., Frank, A.B., and Black, A.L. (1984). Estimation of Spring Wheat Leaf Growth
        Rates and Anthesis from Air Temperature. Agron. J. 76: 829-835] For all crops, there is a base temperature
        under which plant development does not occur (GDD does not accumulate). There is also an upper limit on
        temperature. Past this point, plant development rate cannot increase any further. Corn growth stages are
        divided into vegetative and reproductive stages, and consist of integers preceded by a 'V' or an 'R'.
        Vegetative stages begin with VE (emergence) and end with VT (tasseling) Reproductive stages begin with R1 (
        silking) and end with R6 (physiological maturity) The corn GDD model is defined as follows: [
        https://ndawn.ndsu.nodak.edu/help-corn-growing-degree-days.html] 1. Daily Average Temperature (F) = (Daily
        Max Temp (F) + Daily Min Temp (F)) / 2 2. Daily Corn GDD = Daily Average Temp - 50F 3. Conditions: a. If the
        daily Max and/or Min Temp < 50F, it's set equal to 50F b. If the daily Max Temp > 86F, it's set equal to 86F
        Parameters ------------------------------ start_date_str: (``str``) The start date in the DD/MM/YYYY format
        end_date_str: (``str``) The end date in the DD/MM/YYYY format Returns ------------------------------
        masked_data : (``pd.DataFrame``) The DataFrame containing the changed maximum and minimum temp values and
        also the GDD and accumulated GDD values
        """
        start_date = pd.Timestamp(start_date_str)
        end_date = pd.Timestamp(end_date_str)

        date_mask = (self.daily_data['Date'] >= start_date) & (self.daily_data['Date'] <= end_date)
        masked_data = self.daily_data.loc[date_mask]

        gdd_values = [0]
        agdd_values = [0]
        cumulative_gdd = 0

        # If the daily Max and/or Min Temp < 50 °F (10 °C), it's set equal to 50 °F (10 °C)
        # If the daily Max Temperature > 86 °F (30 °C), it's set equal to 86 °F (30 °C)
        masked_data['Min_temp_F'] = masked_data['Min_temp_F'].apply(lambda x: max(x, 50))
        masked_data['Max_temp_F'] = masked_data['Max_temp_F'].apply(lambda x: max(x, 50))
        masked_data['Max_temp_F'] = masked_data['Max_temp_F'].apply(lambda x: min(x, 86))

        for i in range(1, len(masked_data)):
            mean_temp = (masked_data['Max_temp_F'].iloc[i] + masked_data['Min_temp_F'].iloc[i]) / 2
            gdd = mean_temp - 50

            gdd_values.append(gdd)
            cumulative_gdd += gdd
            agdd_values.append(cumulative_gdd)

        masked_data['gdd'] = gdd_values
        masked_data['agdd'] = agdd_values

        return masked_data

    def growing_degree_days_sugarbeet(self, start_date_str, end_date_str):
        """
        Overview ------------------------------ This function returns a dataframe containing daily and cumulative
        growing degree days (GDD) for sugarbeet. GDD is a measurement of thermal units. Plant growth rate is
        correlated with temperature, so higher temperatures mean faster growth. Based on this concept, GDD can be
        mapped to crop development stage. [Bauer, A., Frank, A.B., and Black, A.L. (1984). Estimation of Spring Wheat
        Leaf Growth Rates and Anthesis from Air Temperature. Agron. J. 76: 829-835] For all crops, there is a base
        temperature under which plant development does not occur (GDD does not accumulate). There is also an upper
        limit on temperature. Past this point, plant development rate cannot increase any further. Beet development
        is quantified using the BBCH scale. This is a continuous scale starting at 0 and ending at 100. [Julius
        Kuhn-Institut (JKI) (2018). Growth stages of mono- and dicotyledonous plants. Julius Kuhn-Institut (JKI).
        DOI: 10.5073/20180906-074619] The sugarbeet GDD model is defined as follows: [
        https://ndawn.ndsu.nodak.edu/help-sugarbeet-growing-degree-days.html] 1. Daily Average Temperature (F) = (
        Daily Max Temp (F) + Daily Min Temp (F)) / 2 2. Daily Sugarbeet GDD = Daily Average Temp - 34F 3. Conditions:
        a. If daily Max or Min Temp < 34F it's set equal to 34F b. If daily Max Temp > 86F it's set equal to 86F
        Parameters ------------------------------ start_date_str: (``str``) The start date in the DD/MM/YYYY format
        end_date_str: (``str``) The end date in the DD/MM/YYYY format Returns ------------------------------
        masked_data : (``pd.DataFrame``) The DataFrame containing the changed maximum and minimum temp values and
        also the GDD and accumulated GDD values
        """
        start_date = pd.Timestamp(start_date_str)
        end_date = pd.Timestamp(end_date_str)

        date_mask = (self.daily_data['Date'] >= start_date) & (self.daily_data['Date'] <= end_date)
        masked_data = self.daily_data.loc[date_mask]

        gdd_values = [0]
        agdd_values = [0]
        cumulative_gdd = 0

        # If daily Max or Min Temp < 34 °F (1.1 °C) it's set equal to 34 °F (1.1 °C)
        # If daily Max Temp > 86 °F (30 °C) it's set equal to 86°F (30 °C)

        masked_data['Max_temp_F'] = masked_data['Max_temp_F'].apply(lambda x: max(x, 34))
        masked_data['Min_temp_F'] = masked_data['Min_temp_F'].apply(lambda x: max(x, 34))
        masked_data['Max_temp_F'] = masked_data['Max_temp_F'].apply(lambda x: min(x, 86))

        for i in range(1, len(masked_data)):
            mean_temp = (masked_data['Max_temp_F'].iloc[i] + masked_data['Min_temp_F'].iloc[i]) / 2
            gdd = mean_temp - 34

            gdd_values.append(gdd)
            cumulative_gdd += gdd
            agdd_values.append(cumulative_gdd)

        masked_data['gdd'] = gdd_values
        masked_data['agdd'] = agdd_values

        return masked_data

    def growing_degree_days_wheat(self, start_date_str, end_date_str):
        """
        Overview
        ------------------------------
        This function returns a dataframe containing daily and cumulative growing degree days (GDD) for wheat.
        GDD is a measurement of thermal units. Plant growth rate is correlated with temperature, so higher
        temperatures mean faster growth. Based on this concept, GDD can be mapped to crop development stage. [Bauer,
        A., Frank, A.B., and Black, A.L. (1984). Estimation of Spring Wheat Leaf Growth Rates and Anthesis from Air
        Temperature. Agron. J. 76: 829-835] For all crops, there is a base temperature under which plant development
        does not occur (GDD does not accumulate). There is also an upper limit on temperature. Past this point,
        plant development rate cannot increase any further.
        For wheat, the Feekes scale is often used to describe growth stages. The Feekes scale begins at 1 (one wheat
        shoot visible) and ends at 11 (grain ripening). [Large, E.C. (1954). Growth stages in cereals - illustration
        of the Feekes scale. Plant Pathol. 3:128-129.] Haun stages are used in wheat GDD calculation because high
        temperatures affect plants differently depending on the growth stage. The Haun stages, being continuous,
        better identify this threshold of when wheat plant growth responds to higher temperatures. While the Feekes
        scale is useful, its discrete nature makes it difficult to discern plant growth stage during vegetative growth.
        The Haun scale consists of a single digit and a single decimal place. The first digit indicates how many
        leaves are fully developed. The decimal indicates the length of the newly developing leaf in comparison to
        the previous fully developed leaf. [Haun, J.R. (1973). Visual quantification of wheat development. Agron. J.
        65:116-119.] The wheat GDD model is defined as follows: [
        https://ndawn.ndsu.nodak.edu/help-wheat-growing-degree-days.html]
        1. Daily Average Temperature (F) = (Daily Max Temp (F) + Daily Min Temp (F)) / 2
        2. Daily Wheat GDD = Daily Average Temp - 32F
        3. Conditions:
            a. If daily Max or Min Temp < 32F it's set equal to 32F
            b. Prior to Haun stage 2.0, (395 GDD accumulated since planting);
                If daily Max Temp > 70F then it's set equal to 70F
            c. After Haun stage 2.0;
                If daily Max Temp > 95F it's set equal to 95F
        Parameters
        ------------------------------
        start_date_str: (``str``)
            The start date in the DD/MM/YYYY format
        end_date_str: (``str``)
            The end date in the DD/MM/YYYY format
        Returns
        ------------------------------
        masked_data: (``pd.DataFrame``)
            The DataFrame containing the changed maximum and minimum temp values
            and also the GDD and accumulated GDD values
        """
        start_date = pd.Timestamp(start_date_str)
        end_date = pd.Timestamp(end_date_str)

        date_mask = (self.daily_data['Date'] >= start_date) & (self.daily_data['Date'] <= end_date)
        masked_data = self.daily_data.loc[date_mask]

        gdd_values = [0]
        agdd_values = [0]

        # If daily Max or daily Min Temp < 32 °F (0 °C) it's set equal to 32 °F (0 °C).
        masked_data.loc[:, 'Min_temp_F'] = masked_data['Min_temp_F'].apply(lambda x: max(x, 32))
        masked_data.loc[:, 'Max_temp_F'] = masked_data['Max_temp_F'].apply(lambda x: max(x, 32))

        cumulative_gdd = 0
        haun_stage = False

        for i in range(1, len(masked_data)):

            if haun_stage:
                masked_data.loc[masked_data.index[i], 'Max_temp_F'] = min(95, masked_data['Max_temp_F'].iloc[i])
            else:
                masked_data.loc[masked_data.index[i], 'Max_temp_F'] = min(70, masked_data['Max_temp_F'].iloc[i])

            mean_temp = (masked_data['Max_temp_F'].iloc[i] + masked_data['Min_temp_F'].iloc[i]) / 2
            gdd = mean_temp - 32

            gdd_values.append(gdd)
            cumulative_gdd += gdd

            haun_stage = cumulative_gdd >= 395
            agdd_values.append(cumulative_gdd)

        masked_data.loc[:, 'gdd'] = gdd_values
        masked_data.loc[:, 'agdd'] = agdd_values

        return masked_data


class ETModel:

    def __init__(self, data, air_temp_col_name, solar_radiation_col_name, timestamp_col_name,
                 barometric_pressure_col_name, windspeed_col_name, rh_col_name, celcius, watts):
        """
        Parameters
        ------------------------------
        data: (``pd.DataFrame``)
            The weather data with hourly readings
        air_temp_col_name: (``str``)
            The name of the air temperature column in the dataframe
        solar_radiation_col_name: (``str``)
            The name of the solar radiation column in the dataframe
        timestamp_col_name: (``str``)
            The name of the timestamp column in the dataframe
        barometric_pressure_col_name: (``str``)
            The name of the barometric pressure column in the dataframe
        windspeed_col_name: (``str``)
            The name of the windspeed column in the dataframe
        rh_col_name: (``str``)
            The name of the relative humidity column in the dataframe
        celcius: (``bool``)
            Indicates whether the temperature is in Celcius units
        watts: (``bool``)
            Indicates whether the solar radiation is in W/m2
        """

        self.timestamp_col_name = timestamp_col_name
        self.solar_radiation_col_name = solar_radiation_col_name
        self.air_temp_col_name = air_temp_col_name
        self.barometric_pressure_col_name = barometric_pressure_col_name
        self.windspeed_col_name = windspeed_col_name
        self.rh_col_name = rh_col_name
        self.watts = watts

        data.loc[:, self.timestamp_col_name] = data[self.timestamp_col_name].apply(lambda x: pd.Timestamp(x))

        if not celcius:
            data.loc[:, self.air_temp_col_name] = data[self.air_temp_col_name].apply(lambda x: (x - 32) / 1.8)

        self.et_data = data

    def get_et(self, start_date_str, end_date_str, os):
        """
        Overview ------------------------------ This function returns reference evapotranspiration (ET) values for a
        user-defined time period. ET refers to the combined processes of plant transpiration and water evaporation
        from the soil surface. [Task Committee on Standardization of Reference Evapotranspiration. (2005). The ASCE
        Standardized Reference Evapotranspiration Equation.]
        Plant transpiration is water evaporation from plant surfaces. ET value therefore indicates how much water
        plants lose over a given period of time. This is useful for crop management because it enables accurate
        scheduling of irrigation. ET is expressed in units of depth, so if a crop has lost 1 inch of water,
        irrigation can be scheduled to replace that water in the soil. Standard practice is to compute a reference
        ET, which quantifies atmospheric 'evaporation power'. Reference ET is then multiplied by a crop coefficient
        to determine crop-specific ET. [Allen, R., Pereira, L., Raes, D., and Smith, M. (1998). Crop
        evapotranspiration: Guidelines for computing crop water requirements. FAO.] The two commonly used reference
        ET models include: 1. Short crop or "os" (corresponds to short grass) 2 Tall crop or "rs" (corresponds to
        full-cover alfalfa)
        The ET formula used here follows the standards that are recommended in the following document:
        https://www.mesonet.org/images/site/ASCE_Evapotranspiration_Formula.pdf
        ET values are calculated at hourly time steps.
        Calculation of utility variables, and page references to the above source can be found in utils.py.
        Parameters
        ------------------------------
        start_date_str: (``str``)
            The start date in the DD/MM/YYYY HH:MM:SS format
        end_date_str: (``str``)
            The end date in the DD/MM/YYYY HH:MM:SS format
        os: (``bool``)
            Indicates whether to calculate ETos or ETrs
        Returns
        ------------------------------
        masked_data: (``pd.DataFrame``)
            The dataframe containing the ET values in in/h for the selected time period
        """

        start_date = pd.Timestamp(start_date_str)
        end_date = pd.Timestamp(end_date_str)
        date_mask = (self.et_data[self.timestamp_col_name] >= start_date) & (
                    self.et_data[self.timestamp_col_name] <= end_date)

        masked_data = self.et_data.loc[date_mask]

        et_values = []

        for i in range(len(masked_data)):

            air_temp = masked_data.loc[masked_data.index[i], self.air_temp_col_name]  # Air temperature

            if self.watts:
                r_n_metric = masked_data.loc[masked_data.index[i], self.solar_radiation_col_name]
                r_n = solar_rad_metric_to_campbell(r_n_metric)
            else:
                r_n = masked_data.loc[masked_data.index[i], self.solar_radiation_col_name]
                r_n_metric = solar_rad_campbell_to_metric(r_n)

            barometric_pressure = masked_data.loc[
                masked_data.index[i], self.barometric_pressure_col_name]  # Air pressure
            rh = masked_data.loc[masked_data.index[i], self.rh_col_name]  # Relative humidity
            wind_speed = masked_data.loc[masked_data.index[i], self.windspeed_col_name]  # Wind speed

            g = get_flux_density(r_n_metric, r_n, os)
            gamma = get_gamma(barometric_pressure)
            cn = get_cn(r_n_metric, os)
            cd = get_cd(r_n_metric, os)
            es = get_es(air_temp)
            ea = get_ea(air_temp, rh)
            delta = get_delta(air_temp)
            u2 = wind_speed

            numerator = 0.408 * delta * (r_n - g)
            numerator += gamma * (cn / (air_temp + 273)) * u2 * (es - ea)
            denominator = delta + gamma * (1 + cd * u2)

            et = numerator / denominator
            et_values.append(et / 25.4)

        et_col_name = 'et'
        if os:
            et_col_name += 'os'
        else:
            et_col_name += 'rs'

        masked_data[et_col_name] = et_values

        return masked_data


class FieldWorkabilityModel:
    """
    The thresholds for this model have not yet been decided. Some resources that can help decide
    the threshold values:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0172301
    https://www.istro.org/index.php/publications/dissertations/61-d15-peter-bilson-obour-soil-workability-and-fragmentation/file
    https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2018MS001477
    https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2018MS001477
    https://www.riverpublishers.com/pdf/ebook/RP_978-87-93237-95-7.pdf
    https://ijoear.com/Paper-June-2018/IJOEAR-APR-2018-7.pdf
    """

    def __init__(self, data, timestamp_col_name, soil_moisture_col_name, air_temp_col_name):

        """
        Parameters
        ------------------------------
        data: (``pd.DataFrame``)
            The weather data with daily readings
        timestamp_col_name: (``str``)
            The name of the timestamp column
        soil_moisture_col_name: (``str``)
            The name of the soil moisture column
        """
        self.data = data
        self.timestamp_col_name = timestamp_col_name
        self.air_temp_col_name = air_temp_col_name
        self.soil_moisture_col_name = soil_moisture_col_name
        self.data[self.timestamp_col_name] = self.data[self.timestamp_col_name].apply(lambda x: pd.Timestamp(x))

    def _get_workability_marker(self, row, sm_threshold_min, sm_threshold_max, temp_threshold_min, temp_threshold_max):

        """
        Internal function not meant to be accessed by the user
        Parameters
        ------------------------------
        row: (``pd.core.series.Series``)
            A single row of data from the dataset
        sm_threshold_min: (``float``)
            The lower threshold for soil moisture
        sm_threshold_max: (``float``)
            The upper threshold for soil moisture
        temp_threshold_min: (``float``)
            The lower threshold for air temperature
        temp_threshold_max: (``float``)
            The upper threshold for air temperature
        Returns
        ------------------------------
        marker: (``bool``)
            A boolean variable indicating whether that particular day is workable
        """
        marker = True

        if row[self.soil_moisture_col_name] <= sm_threshold_min or row[self.soil_moisture_col_name] >= sm_threshold_max:
            marker = False

        if row[self.air_temp_col_name] <= temp_threshold_min or row[self.timestamp_col_name] >= temp_threshold_max:
            marker = False

        return marker

    def num_workable_days(self, sm_threshold_min, sm_threshold_max, temp_threshold_min, temp_threshold_max):
        """
        Parameters
        ------------------------------
        sm_threshold_min: (``float``)
            The lower threshold for soil moisture
        sm_threshold_max: (``float``)
            The upper threshold for soil moisture
        temp_threshold_min: (``float``)
            The lower threshold for air temperature
        temp_threshold_max: (``float``)
            The upper threshold for air temperature
        Returns
        ------------------------------
        workability_results: (``tuple``)
            A tuple containing:
                a) The data containing a boolean column with workable days marked at index 0.
                b) The number of workable days for that time period at index 1.
        """
        self.data['Workable'] = [
            self._get_workability_marker(self.data.iloc[x], sm_threshold_min, sm_threshold_max, temp_threshold_min,
                                         temp_threshold_max)
            for x in range(len(self.data))]

        num_workable_days = self.data['Workable'].sum()

        workability_results = (self.data, num_workable_days)

        return workability_results
