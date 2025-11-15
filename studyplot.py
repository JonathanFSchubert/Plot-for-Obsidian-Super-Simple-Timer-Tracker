import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


def parse_duration(duration_str):
    """Parse a duration string like "1h 23m 40s" into a timedelta object."""
    hours, minutes, seconds = 0, 0, 0
    if "h" in duration_str:
        hours = int(duration_str.split("h")[0])
        duration_str = duration_str.split("h")[1]
    if "m" in duration_str:
        minutes = int(duration_str.split("m")[0])
        duration_str = duration_str.split("m")[1]
    if "s" in duration_str:
        seconds = int(duration_str.split("s")[0])
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def analyze_study_data(file_path):
    """Used the file path from the csv file and analyzes the date by plotting different graphs."""

    # Load the data and parse
    with open(file_path, "r") as file:
        data = file.readlines()

    records = []
    for line in data:
        parts = line.strip().split(",")
        if len(parts) == 4 and parts[1] and parts[2]:
            start_time = datetime.strptime(parts[1], "%y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(parts[2], "%y-%m-%d %H:%M:%S")
            duration = parse_duration(parts[3])
            records.append({
                "Date": start_time.date(),
                "Start Time": start_time,
                "End Time": end_time,
                "Duration": duration
            })

    df = pd.DataFrame(records)

    # Fill in missing dates
    if not df.empty:
        all_dates = pd.date_range(start=df["Date"].min(), end=df["Date"].max())
        study_per_day = df.groupby("Date")["Duration"].sum()
        study_per_day = study_per_day.reindex(all_dates, fill_value=timedelta(0))

    # Convert durations to hours
    study_per_day_hours = study_per_day.apply(lambda x: x.total_seconds() / 3600)

    # Moving average (7-day and 30-day)
    moving_avg_7d = study_per_day_hours.rolling(window=7).mean()
    moving_avg_30d = study_per_day_hours.rolling(window=30).mean()

    # plotting daily study time
    plt.figure(figsize=(14, 8))

    plt.plot(study_per_day_hours, label="Daily Study Time (Hours)", color="skyblue", alpha=0.6)
    plt.plot(moving_avg_7d, label="7-Day Moving Average", color="orange", linewidth=2)
    plt.plot(moving_avg_30d, label="30-Day Moving Average", color="green", linewidth=2)

    plt.title("Study Time Analysis", fontsize=16)
    plt.ylabel("Study Time (Hours)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Create a DataFrame for additional analysis
    study_per_day_df = study_per_day_hours.reset_index()
    study_per_day_df.columns = ["Date", "Study Hours"]
    study_per_day_df["Month"] = study_per_day_df["Date"].dt.month
    study_per_day_df["Day"] = study_per_day_df["Date"].dt.day
    study_per_day_df["Weekday"] = study_per_day_df["Date"].dt.weekday
    study_per_day_df["YearMonth"] = study_per_day_df["Date"].dt.to_period("M")

    # Heatmap: Average Study Time by Weekday and Month
    weekday_month_avg = study_per_day_df.groupby(["Weekday", "YearMonth"])["Study Hours"].mean().unstack()

    plt.figure(figsize=(14, 8))
    sns.heatmap(weekday_month_avg, cmap="Blues", annot=True, fmt=".1f", cbar=True)
    plt.title("Average Study Time by Weekday and Month", fontsize=16)
    plt.xlabel("Month")
    plt.ylabel("Weekday")
    plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    plt.tight_layout()
    plt.show()


    # Study Density by Weekday & Time-of-Day (last X months)

    # Filter to last x months
    x_months = 4
    today = datetime.now()
    x_months_ago = today - pd.DateOffset(months=x_months)
    df_recent = df[df["Start Time"] >= x_months_ago]

    # Prepare a 7 × 48 matrix (weekdays × half-hour bins)
    bins = 48
    heat = pd.DataFrame(0, index=range(7), columns=range(bins), dtype=float)

    # Accumulate minute-by-minute
    for _, row in df_recent.iterrows():
        start_time, end_time = row["Start Time"], row["End Time"]
        # handle sessions crossing midnight
        while start_time < end_time:
            # end of current day
            day_end = datetime(start_time.year, start_time.month, start_time.day, 23, 59, 59)
            seg_end = min(end_time, day_end)
            # walk minute by minute
            cursor = start_time
            while cursor <= seg_end:
                weekday = cursor.weekday()  # 0=Mon,6=Sun
                minute_of_day = cursor.hour*60 + cursor.minute
                bin_idx = minute_of_day // 30
                heat.at[weekday, bin_idx] += 1
                cursor += timedelta(minutes=1)
            # move to next day
            start_time = seg_end + timedelta(seconds=1)

    # Convert counts to hours
    heat = heat / 60.0

    # Convert to average study time per half-hour bin
    weeks = (df_recent["Start Time"].max() - df_recent["Start Time"].min()).days / 7
    if weeks > 0:
        heat = heat / weeks

    # Convert hours to minutes
    heat *= 60

    # Plot the study time density
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        heat,
        cmap="plasma",
        cbar_kws={"label": "Average Study Minutes"},
        linewidths=0.5, linecolor="gray"
    )
    plt.title(f"Study Time Density by Weekday & Half-Hour (Last {x_months} Months)", fontsize=16)
    plt.ylabel("Weekday")
    plt.yticks(
        ticks=[0,1,2,3,4,5,6],
        labels=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        rotation=0
    )
    # Label every 2 hours on x-axis
    xticks = list(range(0, bins, 4))
    xlabels = [f"{h:02d}:00" for h in range(0, 24, 2)]
    plt.xticks(xticks, xlabels, rotation=45)
    plt.xlabel("Time of Day")
    plt.tight_layout()
    plt.show()


# file path to the CSV file
file_path = "studytime.csv"
analyze_study_data(file_path)

