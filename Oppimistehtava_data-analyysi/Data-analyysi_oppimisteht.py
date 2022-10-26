import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("202209-citibike-tripdata.csv")

# Bike dataframe
df_bikes = df['rideable_type'].value_counts().div(pow(10,6))
df_bikes = df_bikes.reset_index()
df_bikes = df_bikes.rename(columns={"index":"bike_type",
                                    "rideable_type":"bike_count"})
# df_bikes = df_bikes.loc[:,['bike_count']].div(pow(10,6))
# Start point dataframe
df_start = df['start_station_name'].value_counts()
df_start = df_start.reset_index()
df_start = df_start.rename(columns={"index":"start_station",
                                    "start_station_name":"station_count"})

# start times converted from datetime to only date
df_start_times = pd.to_datetime(df['started_at']).dt.date
df_start_times = df_start_times.value_counts()
# same than start
df_end_times = pd.to_datetime(df['ended_at']).dt.date
df_end_times = df_end_times.value_counts()

# Member types
df_members = df['member_casual'].value_counts()
df_members = df_members.reset_index()
df_members = df_members.rename(columns={"index":"member_type",
                                        "member_casual":"member_count"})

# Inserted start and end times to new dataframe
df_times = pd.DataFrame({'started_date':df_start_times,
                          'ended_date':df_end_times})
# df_times = df_times.value_counts()
df_times = df_times.reset_index()
df_times = df_times.rename(columns={"index":"date","started_date":"started",
                                    "ended_date":"ended"})

df_bikes.plot.bar(x='bike_type',rot=0)
plt.show()
# df_times.plot.line()
# plt.show()