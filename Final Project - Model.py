
import pandas as pd
from ortools.sat.python import cp_model

# Load the Excel files
path = '/Users/admin/PycharmProjects/FinalProject/קבצי המודל'
df_raw = pd.read_excel(f'{path}/lessons.xlsx')
df_buildings = pd.read_excel(f'{path}/buildings_faculties_names.xlsx')

faculty_name = 'ביה"ס לחינוך'
df_buildings_filtered = df_buildings[df_buildings['שם הפקולטה'] == faculty_name]
valid_buildings = df_buildings_filtered['שם הבניין'].unique()

# Prepare lessons data with correct filtering
df_prepare = df_raw[
    (df_raw["חדר"] != "0") &
    (df_raw["סמסטר"] == 1) &
    (df_raw["שנתי"] == "לא") &
    (df_raw["נלווה"] != "נ") &
    (df_raw["תיאור סוג זמק"] == "שבועי רגיל") &
    (df_raw["קיבולת חדר"] != 0) &
    (df_raw["שם_פקולטה"] == faculty_name)
]

# Define the desired columns
desired_columns = [
    'שם_פקולטה', 'שם_ביה"ס', 'שם_חוג', 'מס_קורס', 'הקבצה', 'שם_קורס',
    'אופן הוראה', 'תיאור אופן הוראה', 'מספר תלמידים (צפי)', 'מספר תלמידים (רשומים)',
    'קוד מטלה ראשי', 'קודי מטלות נוספים', 'סוג זמ"ק', 'תיאור סוג זמק',
    'שעת_התחלה', 'שעת_סיום', 'יום_בשבוע', 'חדר', 'שם הבניין', 'קיבולת חדר',
    'הערה פנימית למקצוע', 'הערה חיצונית למקצוע'
]

df = df_prepare.loc[:, desired_columns]  # Use .loc to select columns

# Drop duplicate rows
df = df.drop_duplicates()

# Preprocess time columns to ensure they are in the correct format
def preprocess_time(time_str):
    if pd.isna(time_str):
        return time_str
    time_str = str(time_str).strip()
    if len(time_str) == 1:
        return f"0{time_str}:00"
    elif len(time_str) == 2:
        return f"{time_str}:00"
    elif len(time_str) == 3:
        return f"0{time_str[:1]}:{time_str[1:]}0"
    elif len(time_str) == 4:
        return f"{time_str[:2]}:{time_str[2:]}0"
    elif len(time_str) == 5 and time_str.count(':') == 1:
        return time_str
    else:
        raise ValueError(f"Unexpected time format: {time_str}")

df['שעת_התחלה'] = df['שעת_התחלה'].apply(preprocess_time)
df['שעת_סיום'] = df['שעת_סיום'].apply(preprocess_time)

# Specify the time format explicitly
time_format = '%H:%M'
try:
    df['שעת_התחלה'] = pd.to_datetime(df['שעת_התחלה'], format=time_format).dt.time
    df['שעת_סיום'] = pd.to_datetime(df['שעת_סיום'], format=time_format).dt.time
except ValueError as e:
    print(f"Error in time conversion: {e}")

# Create a unique identifier combining few parameters
df['unique_id'] = (
    df['מס_קורס'].astype(str) + '_' +
    df['הקבצה'].astype(str) + '_' +
    df['שעת_התחלה'].astype(str) + '_' +
    df['שעת_סיום'].astype(str) + '_' +
    df['יום_בשבוע'].astype(str) + '_'
)

# Sort lessons by number of students in descending order
df = df.sort_values(by='מספר תלמידים (רשומים)', ascending=False)

unique_ids = df['unique_id'].unique()
classroom_ids = df['חדר'].unique()

# Display some information
print(f'unique_id size is: {unique_ids.size}')
print(f'classroom_ids size is: {classroom_ids.size}')
print(df.head())
print(df.shape)

# Count the number of unique classrooms used, broken down by faculty affiliation
classrooms_used = df['חדר'].unique()
classrooms_in_faculty = df[df['שם הבניין'].isin(valid_buildings)]['חדר'].unique()
classrooms_not_in_faculty = df[~df['שם הבניין'].isin(valid_buildings)]['חדר'].unique()

# Calculate the over-capacity lessons
out_of_range_lessons = (df['קיבולת חדר'] < df['מספר תלמידים (רשומים)']).sum()

# Print summary before the model starts
print('Summary for input schedule - Before running the model:')
print(f'Total lessons:\n{df.shape[0]}')
print(f'Total unique classrooms used:\n{len(classrooms_used)}')
print(f'The number of classes assigned to classrooms that exceed their capacity:\n{out_of_range_lessons}')


print('\nFinished preparing the data.')
# Initialize the CP-SAT model
model = cp_model.CpModel()
print('Start running the model')

# Decision variable: lesson in a classroom, boolean Var - if the lesson scheduled in that time in that class, yes/no.
schedule = {}
for unique_id in unique_ids:
    for classroom in classroom_ids:
        schedule[(unique_id, classroom)] = model.NewBoolVar(f'schedule_{unique_id}_{classroom}')

# Constraint to ensure all lessons are scheduled exactly once
for unique_id in unique_ids:
    model.Add(sum(schedule[(unique_id, classroom)] for classroom in classroom_ids) == 1)

# Classroom capacity constraints
for idx, row in df.iterrows():
    unique_id = row['unique_id']
    num_students = row['מספר תלמידים (רשומים)']
    for classroom in classroom_ids:
        room_capacity = df[df['חדר'] == classroom]['קיבולת חדר'].values[0]
        if room_capacity < num_students:
            model.Add(schedule[(unique_id, classroom)] == 0)

# No scheduling conflicts
for classroom in classroom_ids:
    for day in df['יום_בשבוע'].unique():
        for time in df['שעת_התחלה'].unique():
            model.Add(
                sum(
                    schedule[(row['unique_id'], classroom)]
                    for idx, row in df.iterrows()
                    if row['יום_בשבוע'] == day and row['שעת_התחלה'] == time
                ) <= 1
            )

# Objective function to minimize changes from original schedule
original_schedule = {(row['unique_id'], row['חדר']): 1 for idx, row in df.iterrows()}
objective_terms = []
for unique_id in unique_ids:
    for classroom in classroom_ids:
        if (unique_id, classroom) in original_schedule:
            objective_terms.append(schedule[(unique_id, classroom)])

model.Maximize(sum(objective_terms))

# Solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Print the solution and the changes made
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    scheduled_lessons = []
    classrooms_used = set()
    classrooms_in_faculty_used = set()
    classrooms_not_in_faculty_used = set()
    for unique_id in unique_ids:
        for classroom in classroom_ids:
            if solver.Value(schedule[(unique_id, classroom)]):
                # Get course details for the unique_id
                row = df[df['unique_id'] == unique_id].iloc[0]
                course_name = row['שם_קורס']
                start_time = row['שעת_התחלה']
                end_time = row['שעת_סיום']
                day = row['יום_בשבוע']
                building = row['שם הבניין']
                capacity = row['קיבולת חדר']
                number_of_signed_students = row['מספר תלמידים (רשומים)']
                scheduled_lessons.append((unique_id, course_name, classroom, day, start_time, end_time, building, capacity, number_of_signed_students))
                classrooms_used.add(classroom)
                if building in valid_buildings:
                    classrooms_in_faculty_used.add(classroom)
                else:
                    classrooms_not_in_faculty_used.add(classroom)

    # Summarize changes
    total_lessons = len(df)
    scheduled_count = len(scheduled_lessons)
    unscheduled_count = total_lessons - scheduled_count
    print(f'\nSummary of Changes - After running the model:')
    print(f'Total lessons:\n{total_lessons}')
    print(f'Number of lessons scheduled:\n{scheduled_count}')
    print(f'Total unique classrooms used:\n{len(classrooms_used)}')


    # Calculate the percentage of scheduling changes
    initial_schedule = df[['unique_id','שם_קורס', 'חדר', 'יום_בשבוע', 'שעת_התחלה', 'שעת_סיום', 'שם הבניין', 'קיבולת חדר', 'מספר תלמידים (רשומים)']].copy()
    initial_schedule['scheduled'] = False
    initial_schedule['new_classroom'] = None
    initial_schedule['new_building'] = None
    initial_schedule['new_capacity'] = None

    for lesson in scheduled_lessons:
        unique_id, course_name, classroom, day, start_time, end_time, building, capacity, number_of_signed_students = lesson
        initial_schedule.loc[
            (initial_schedule['unique_id'] == unique_id), ['scheduled', 'new_classroom', 'new_building', 'new_capacity']] = [True, classroom, building, df[df['חדר'] == classroom]['קיבולת חדר'].values[0]]

    changes_count = (initial_schedule['חדר'] != initial_schedule['new_classroom']).sum()
    changes_percentage = (changes_count / total_lessons) * 100
    print(f'Number of leassons that changed classrooms:\n{changes_count}')
    print(f'Percentage of scheduling changes:\n{changes_percentage:.2f}%')

    # Filter the DataFrame for only lessons where the classroom has changed
    df_changes = initial_schedule[initial_schedule['חדר'] != initial_schedule['new_classroom']]
    df_changes = df_changes.rename(columns={'חדר': 'original_classroom', 'שם הבניין': 'original_building', 'קיבולת חדר': 'original_capacity'})
    df_changes = df_changes[['unique_id','שם_קורס', 'מספר תלמידים (רשומים)', 'שעת_התחלה', 'שעת_סיום', 'יום_בשבוע', 'original_classroom', 'original_capacity', 'original_building', 'new_classroom', 'new_capacity', 'new_building']]

    # Save the changes to an Excel file
    output_path = f'{path}/ results - {faculty_name}.xlsx'
    df_changes.to_excel(output_path, index=False)

    print(f'\nChanges saved to {output_path}')

else:
    print('A change in schedule cannot be made with the existing resources.')

