import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("202209-citibike-tripdata.csv")

# Bike dataframe
df_bikes = df['rideable_type'].value_counts().div(pow(10,6))
df_bikes = df_bikes.reset_index()
df_bikes = df_bikes.rename(columns={"index":"bike_type",
                                    "rideable_type":"bike_count"})
# df_user_type = pd.get_dummies(df['member_casual'],drop_first=True)
# Start point dataframe
df_start = df['start_station_name'].value_counts()
df_start = df_start.reset_index()
df_start = df_start.rename(columns={"index":"start_station",
                                    "start_station_name":"station_count"})

# start dates converted from datetime to only date
df_start_date = pd.to_datetime(df['started_at']).dt.date
df_start_date = df_start_date.value_counts()

# end dates, same conversion than start
df_end_date = pd.to_datetime(df['ended_at']).dt.date
df_end_date = df_end_date.value_counts()

# Ride durations
df_start_time = pd.to_datetime(df['started_at']).dt.time
df_end_time = pd.to_datetime(df['ended_at']).dt.time
df_ride_time = pd.DataFrame({'start_time':df_start_time,
                             'end_time':df_end_time})

# Member types
df_users = df['member_casual'].value_counts()
df_users = df_users.reset_index()
df_users = df_users.rename(columns={"index":"user_type",
                                        "member_casual":"user_count"})

# Inserted start and end times to new dataframe
df_times = pd.DataFrame({'started_date':df_start_date,
                          'ended_date':df_end_date})
# df_times = df_times.value_counts()
df_times = df_times.reset_index()
df_times = df_times.rename(columns={"index":"date","started_date":"started",
                                    "ended_date":"ended"})

df_bikes.plot.bar(x='bike_type',rot=0)
plt.show()
# users = df_users.plot.pie(y='')
# plt.show()
# df_times.plot.line()
# plt.show()

