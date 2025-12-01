import pyspark.sql.functions as sf
from pyspark.sql import SparkSession, Window
from columns import event_names, date_columns

"""
Data Formats:
 - HPC: RegionID,SizeRank,RegionName,RegionType,StateName,
        State,Metro,StateCodeFIPS,MunicipalCodeFIPS,
        2000-01-31,2000-02-29,2000-03-31,2000-04-30,
        ...,
        2025-10-31
        (quarterly data for 2000-2025)
 - NOAA:
    - avg_temperature: fips,date,year,month,avg_temp_f
    - max_temperature: fips,date,year,month,max_temp_f
    - min_temperature: fips,date,year,month,min_temp_f
    - precipitation: fips,date,year,month,precip_inches
 - FEMA:
    - NRI_HazardInfo: OID_,Hazard,Prefix,Service,Start,
                      End_,TotalYears,FrequencyModel,
                      EAL_Building,EAL_Population,
                      EAL_Agriculture,PeriodOfRecord,
                      NRI_VER
    - NRI_Table_Counties: OID_,NRI_ID,STATE,STATEABBRV,
                          STATEFIPS,COUNTY,COUNTYTYPE,
                          COUNTYFIPS,STCOFIPS,POPULATION,
                          BUILDVALUE,AGRIVALUE,AREA,
                          RISK_VALUE,RISK_SCORE,RISK_RATNG,
                          RISK_SPCTL,EAL_SCORE,EAL_RATNG,
                          EAL_SPCTL,EAL_VALT,EAL_VALB,
                          EAL_VALP,EAL_VALPE,EAL_VALA,
                          ALR_VALB,ALR_VALP,ALR_VALA,
                          ALR_NPCTL,ALR_VRA_NPCTL,
                          SOVI_SCORE,SOVI_RATNG,SOVI_SPCTL,
                          RESL_SCORE,RESL_RATNG,RESL_SPCTL,
                          RESL_VALUE,CRF_VALUE,AVLN_EVNTS,
                          ...
    - NRIDataDictionary: Sort,Field Name,Field Alias,Type,Length,Relevant Layer,Metric Type,Version,Version Date

Expected Output Schema:

    8 Climate Features + 1 Target Variable

    Feature                    Unit        Source

    1. avg_temperature         °F          NOAA
    2. max_temperature         °F          NOAA  
    3. min_temperature         °F          NOAA
    4. precipitation           inches      NOAA
    5. sea_level_change        mm          NOAA
    6. extreme_weather_count   count       NRI/FEMA
    7. drought_severity        0-10        NRI/FEMA
    8. flood_risk_score        0-100       NRI/FEMA

    TARGET: price_change       %           Zillow/Redfin
"""

def create_spark_session():
    spark = (SparkSession.builder
         .appName("Preprocessing")
         .master("local")
         .getOrCreate())
    return spark

def load_data(spark):
    nri_hazard_info = spark.read.csv("Data/Fema/NRI_Table_Counties/NRI_HazardInfo.csv", header=True, inferSchema=True)
    nri_table_counties = spark.read.csv("Data/Fema/NRI_Table_Counties/NRI_Table_Counties.csv", header=True, inferSchema=True)
    nri_data_dictionary = spark.read.csv("Data/Fema/NRI_Table_Counties/NRIDataDictionary.csv", header=True, inferSchema=True)
    noaa_avg_temperature = spark.read.csv("Data/NOAA/avg_temperature.csv", header=True, inferSchema=True)
    noaa_max_temperature = spark.read.csv("Data/NOAA/max_temperature.csv", header=True, inferSchema=True)
    noaa_min_temperature = spark.read.csv("Data/NOAA/min_temperature.csv", header=True, inferSchema=True)
    noaa_precipitation = spark.read.csv("Data/NOAA/precipitation.csv", header=True, inferSchema=True)
    hpc_data = spark.read.csv("Data/HPC/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", header=True, inferSchema=True)

    return {
        "nri_hazard_info": nri_hazard_info,
        "nri_table_counties": nri_table_counties,
        "nri_data_dictionary": nri_data_dictionary,
        "noaa_avg_temperature": noaa_avg_temperature,
        "noaa_max_temperature": noaa_max_temperature,
        "noaa_min_temperature": noaa_min_temperature,
        "noaa_precipitation": noaa_precipitation,
        "hpc_data": hpc_data
    }

# get all the relevant NOAA data for each county for each month
# use left joins for all of these since we want to keep what we have and specify what to add on top
def join_noaa_data(data_dict):
    climate_df = data_dict['noaa_avg_temperature'].select(
        'fips',
        'year',
        'month',
        'avg_temp_f'
    )

    climate_df = climate_df.join(
        data_dict['noaa_max_temperature'].select('fips', 'year', 'month', 'max_temp_f'),
        on=['fips', 'year', 'month'],
        how='left'
    )

    climate_df = climate_df.join(
        data_dict['noaa_min_temperature'].select('fips', 'year', 'month', 'min_temp_f'),
        on=['fips', 'year', 'month'],
        how='left'
    )

    climate_df = climate_df.join(
        data_dict['noaa_precipitation'].select('fips', 'year', 'month', 'precip_inches'),
        on=['fips', 'year', 'month'],
        how='left'
    )

    return climate_df

def get_ew_counts(data_dict):
    nri_df = data_dict['nri_table_counties'].select(
        sf.col('STCOFIPS').alias('fips'),
        *event_names
    )
    nri_df = nri_df.withColumn('total_events', sum([sf.coalesce(sf.col(x), sf.lit(0)) for x in event_names]).cast('int'))
    nri_df = nri_df.select('fips', 'total_events')

    return nri_df

def get_drought_severity(data_dict):
    drought_df = data_dict['nri_table_counties'].select(
        sf.col('STCOFIPS').alias('fips'),
        sf.col('DRGT_AFREQ').alias('drought_severity')
    )
    drought_df = drought_df.select('fips', 'drought_severity')
    return drought_df

def get_flood_risk_score(data_dict):
    flood_df = data_dict['nri_table_counties'].select(
        sf.col('STCOFIPS').alias('fips'),
        sf.col('RFLD_RISKS').alias('flood_risk_score')
    )
    flood_df = flood_df.select('fips', 'flood_risk_score')
    return flood_df

def get_price_change(data_dict):
    hpc_df = data_dict['hpc_data']
    housing_df = hpc_df.withColumn(
        'fips',
        sf.concat(
            sf.lpad(sf.col('StateCodeFIPS'), 2, '0'),
            sf.lpad(sf.col('MunicipalCodeFIPS'), 3, '0')
        )
    )
    housing_df = housing_df.select('fips', *date_columns)

    col_names = []
    for i in range(1, len(date_columns)):
        current_col = date_columns[i]
        prev_col = date_columns[i-1]

        current_year = current_col.split('-')[0]
        current_month = current_col.split('-')[1]
        prev_year = prev_col.split('-')[0]
        prev_month = prev_col.split('-')[1]

        if current_month == prev_month and current_year == prev_year:
            continue
        
        col_name = f'{current_year}_{current_month}'
        col_names.append(col_name)

        price_change = ((sf.col(current_col) - sf.col(prev_col)) / sf.col(prev_col)) * 100

        housing_df = housing_df.withColumn(
            col_name,
            price_change
        )
    
    housing_df = housing_df.drop(*date_columns)
    num_cols = len(col_names)
    stack_string = ', '.join([f"'{col.split('_')[0]}', '{col.split('_')[1]}', {col}" for col in col_names])
    
    housing_df = housing_df.select(
        'fips',
        sf.expr(f"stack({num_cols}, {stack_string}) as (year, month, price_change)")
    )

    # convert to int since noaa date in int format
    housing_df = housing_df.withColumn('year', sf.col('year').cast('int'))
    housing_df = housing_df.withColumn('month', sf.col('month').cast('int'))

    return housing_df

def join_all_data(noaa_df, nri_df, drought_df, flood_df, housing_df):
    output_df = noaa_df.join(nri_df, on='fips', how='left')
    output_df = output_df.join(drought_df, on='fips', how='left')
    output_df = output_df.join(flood_df, on='fips', how='left')
    output_df = output_df.join(housing_df, on=['fips', 'year', 'month'], how='left')
    return output_df

def main():
    spark = create_spark_session()
    data_dict = load_data(spark)
    noaa_df = join_noaa_data(data_dict)
    nri_df = get_ew_counts(data_dict)
    drought_df = get_drought_severity(data_dict)
    flood_df = get_flood_risk_score(data_dict)
    housing_df = get_price_change(data_dict)
    output_df = join_all_data(noaa_df, nri_df, drought_df, flood_df, housing_df)
    output_df.show(20)
    print(f"Total rows: {output_df.count()}")

if __name__ == "__main__":
    main()