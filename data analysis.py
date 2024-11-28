import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import matplotlib.ticker as ticker  # 시간 형식 지정

# Seaborn 스타일 설정
sns.set(style="whitegrid")

# 프레임 속도 설정
frame_rate = 30.0  # 30fps

# 결과 디렉토리 경로 (필요에 따라 수정)
results_dir = r'C:\Users\USER\Desktop\Videoframe-Image-saver-main\output\20241120-143811-185(0.55_3_3d)\Results'  # 실제 경로로 업데이트

# 사용자 입력을 받아 검증하는 함수
def get_user_input(prompt, default, is_float=False, min_value=None, max_value=None):
    while True:
        try:
            user_input = input(prompt)
            if user_input.strip() == '':
                return default
            value = float(user_input) if is_float else int(user_input)
            if value == 0:
                return default
            if min_value is not None and value < min_value:
                print(f"Value cannot be less than {min_value}. Please try again.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value cannot be greater than {max_value}. Please try again.")
                continue
            return value
        except:
            print("Invalid input. Please enter a valid number.")
            continue

# 사용자로부터 농도(phi) 입력 받기
phi = get_user_input(
    "Enter concentration value (φ) [default: 0.45]: ",
    0.45, is_float=True, min_value=0.0
)
print(f"User-input concentration (φ): {phi}")

# 모든 결과 CSV 파일 수집
results_files = glob.glob(os.path.join(results_dir, "*_results.csv"))

# 모든 CSV 파일을 읽어 하나의 DataFrame으로 합치기
df_list = []
for file in results_files:
    # 파일명에서 이미지 번호 추출
    filename = os.path.basename(file)
    image_number_str = ''.join(filter(str.isdigit, filename))

    # 이미지 번호가 없는 파일은 건너뜀
    if image_number_str == '':
        print(f"Image number not found in filename: {filename}. Skipping this file.")
        continue

    image_number = int(image_number_str)

    # CSV 파일 읽기
    try:
        temp_df = pd.read_csv(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    temp_df['ImageNumber'] = image_number
    df_list.append(temp_df)

# 유효한 데이터가 있는지 확인
if not df_list:
    raise ValueError("No valid CSV files found or all CSV files are empty.")

# 모든 데이터를 하나의 DataFrame으로 결합
all_particles_df = pd.concat(df_list, ignore_index=True)

# ImageNumber 기준으로 정렬
all_particles_df = all_particles_df.sort_values('ImageNumber')

# 초기 및 최대 이미지 번호 설정
initial_image_number = all_particles_df['ImageNumber'].min()
max_image_number = all_particles_df['ImageNumber'].max()

# 사용 가능한 ImageNumber 범위 출력
print(f"Available ImageNumber range: {initial_image_number} to {max_image_number}")

# 산점도를 위한 이미지 범위와 간격 사용자 입력 받기
print("Please enter the image range for scatter plots.")
start_image_number = get_user_input(
    f"Enter start ImageNumber (default: 120, min: {initial_image_number}, max: {max_image_number}): ",
    120, is_float=False, min_value=initial_image_number, max_value=max_image_number)
end_image_number = get_user_input(
    f"Enter end ImageNumber (default: 170, min: {start_image_number}, max: {max_image_number}): ",
    170, is_float=False, min_value=start_image_number, max_value=max_image_number)
interval = get_user_input("Enter image number interval (default: 10): ", 10, is_float=False, min_value=1)

# 축 스케일 팩터 사용자 입력 받기
x_scale_factor = get_user_input("Enter x-axis scale factor (default: 1.05): ", 1.05, is_float=True)
y_scale_factor = get_user_input("Enter y-axis scale factor (default: 1.05): ", 1.05, is_float=True)

# 픽셀을 밀리미터로 변환하기 위한 변환 계수 정의
pixels_to_mm = 10 / 2238  # 1 px = 10 / 2238 mm ≈ 0.004467 mm/px
pixels2_to_mm2 = pixels_to_mm ** 2  # 1 px² = (10 / 2238)² mm² ≈ 0.0000199 mm²/px²

# 중심점 (3840, 2160)으로부터 각 입자의 거리 계산
if 'X' in all_particles_df.columns and 'Y' in all_particles_df.columns:
    all_particles_df['Distance'] = np.sqrt((all_particles_df['X'] - 3840) ** 2 + (all_particles_df['Y'] - 2160) ** 2)
    # 거리(mm)으로 변환
    all_particles_df['Distance to Center (mm)'] = all_particles_df['Distance'] * pixels_to_mm
else:
    print("Error: 'X' and/or 'Y' columns are missing in the data.")
    raise ValueError("Required columns 'X' and 'Y' are missing.")

# 면적(px²)을 mm²로 변환
if 'Area' in all_particles_df.columns:
    all_particles_df['Particle Area (mm^2)'] = all_particles_df['Area'] * pixels2_to_mm2
else:
    print("Error: 'Area' column is missing in the data.")
    raise ValueError("Required column 'Area' is missing.")

# 시작 이미지 번호를 기준으로 시간을 초 단위로 계산
all_particles_df['Time'] = ((all_particles_df['ImageNumber'] - start_image_number) / frame_rate).round(2)

# 산점도를 위한 이미지 범위와 간격에 따라 데이터 필터링
selected_df = all_particles_df[
    (all_particles_df['ImageNumber'] >= start_image_number) &
    (all_particles_df['ImageNumber'] <= end_image_number) &
    (all_particles_df['ImageNumber'] % interval == 0)
]

# 시간에 따른 입자 수 계산
particle_count_over_time = selected_df.groupby('Time').size().reset_index(name='ParticleCount')

# 시간에 따른 입자 수 그래프 그리기
plt.figure(figsize=(12, 6))
sns.lineplot(data=particle_count_over_time, x='Time', y='ParticleCount', marker='o')
plt.title(f'Number of Particles Over Time (φ={phi})', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Particle Count', fontsize=14)

# x축 형식 지정
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()

# 입자 수 그래프 저장
plots_dir = os.path.join(os.path.dirname(results_dir), 'Plots')
os.makedirs(plots_dir, exist_ok=True)

particle_count_save_path = os.path.join(plots_dir, f'Number_of_Particles_Over_Time_φ={phi}.png')
plt.savefig(particle_count_save_path, dpi=300)
print(f"Saved particle count graph to: {particle_count_save_path}")
plt.close()

# 시간에 따른 총 면적 계산
total_area_over_time = selected_df.groupby('Time')['Particle Area (mm^2)'].sum().reset_index(name='TotalArea_mm2')

# 시간에 따른 총 면적 그래프 그리기
plt.figure(figsize=(12, 6))
sns.lineplot(data=total_area_over_time, x='Time', y='TotalArea_mm2', marker='o', color='green')
plt.title(f'Total Particle Area Over Time (φ={phi})', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Total Area (mm²)', fontsize=14)

# x축 형식 지정
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()

# 총 면적 그래프 저장
total_area_save_path = os.path.join(plots_dir, f'Total_Particle_Area_Over_Time_φ={phi}.png')
plt.savefig(total_area_save_path, dpi=300)
print(f"Saved total area graph to: {total_area_save_path}")
plt.close()

# 'Distance to Center (mm)' 컬럼이 존재할 경우 계속 진행
if 'Distance to Center (mm)' in all_particles_df.columns:
    # 거리(mm) 기준으로 빈 정의
    bin_size_px = 50  # 원래 픽셀 단위 빈 크기
    bin_size_mm = bin_size_px * pixels_to_mm  # mm 단위로 변환
    max_distance_mm = all_particles_df['Distance to Center (mm)'].max()
    distance_bins_mm = np.arange(0, max_distance_mm + bin_size_mm, bin_size_mm)
    all_particles_df['DistanceBin_mm'] = pd.cut(all_particles_df['Distance to Center (mm)'], bins=distance_bins_mm)

    # 산점도에 선택된 시간 추출
    unique_times = selected_df['Time'].unique()
    unique_times = np.round(unique_times, 2)
    unique_times = sorted(unique_times)
    print(f"Unique Times for plotting: {unique_times}")

    # 산점도를 위한 샘플링 비율 사용자 입력 받기
    sampling_rate = get_user_input(
        "Enter sampling rate for scatter plots (between 0 and 1, default: 1.0 for no sampling): ",
        1.0, is_float=True, min_value=0.0, max_value=1.0
    )
    print(f"Sampling rate: {sampling_rate}")

    # 샘플링 비율이 1.0 미만일 경우 데이터 샘플링 적용
    if sampling_rate < 1.0:
        # 그룹별 샘플링을 사용하여 DeprecationWarning 방지
        sampled_df = selected_df.groupby('Time', group_keys=False).sample(
            frac=sampling_rate, random_state=42
        )
        print(f"Sampled selected_df to {sampling_rate*100:.0f}% of data for plotting.")
    else:
        sampled_df = selected_df.copy()

    # 샘플링된 데이터에 'Distance to Center (mm)' 컬럼이 있는지 확인
    if 'Distance to Center (mm)' not in sampled_df.columns:
        print("Error: 'Distance to Center (mm)' column is missing in the sampled data.")
        print(f"Available columns: {sampled_df.columns.tolist()}")
        raise ValueError("Required column 'Distance to Center (mm)' is missing after sampling.")

    print(f"Number of particles before sampling: {len(selected_df)}")
    print(f"Number of particles after sampling: {len(sampled_df)}")

    # 고유 시간 수에 맞춰 색상 팔레트 생성
    colors = sns.color_palette('viridis', n_colors=len(unique_times))

    # 산점도 그리기: 입자 면적(mm²) vs. 거리(mm), 시간별 색상 구분
    plt.figure(figsize=(14, 10))

    scatter = sns.scatterplot(
        data=sampled_df,
        x='Distance to Center (mm)',
        y='Particle Area (mm^2)',
        hue='Time',
        palette='viridis',
        alpha=0.6,
        edgecolor=None,
        legend=False  # 기본 범례 비활성화
    )

    # 사용자 정의 범례 핸들 및 라벨 생성
    handles = []
    labels = []
    for idx, time_value in enumerate(unique_times):
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color=colors[idx]))
        labels.append(f"{time_value:.2f}s")

    # 범례의 열 수 결정
    num_unique_times = len(unique_times)
    if num_unique_times > 8:
        legend_ncol = 2
    else:
        legend_ncol = 1

    # 범례 추가
    plt.legend(
        handles,
        labels,
        title='Time (s)',
        fontsize=12,
        title_fontsize=14,
        loc='upper left',
        bbox_to_anchor=(1, 1),
        ncol=legend_ncol,
        frameon=False
    )

    # 축 스케일 팩터 적용
    overall_distance_min = 0
    overall_distance_max = all_particles_df['Distance to Center (mm)'].max()
    overall_area_min = 0
    overall_area_max = all_particles_df['Particle Area (mm^2)'].max()

    plt.xlim(overall_distance_min, overall_distance_max * x_scale_factor)
    plt.ylim(overall_area_min, overall_area_max * y_scale_factor)

    # x축 형식 지정
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 샘플링 비율 주석 추가
    plt.text(
        0.95, 0.95, f'Sampling Rate: {sampling_rate * 100:.0f}%',
        horizontalalignment='right',
        verticalalignment='top',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
    )
    plt.title(f'Particle Area vs. Distance at Selected Times (φ={phi})', fontsize=18)
    plt.grid(True)
    plt.tight_layout()

    # 산점도 저장
    scatter_save_path = os.path.join(plots_dir, f'Particle_Area_vs_Distance_Selected_Time_φ={phi}.png')
    plt.savefig(scatter_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to: {scatter_save_path}")
    plt.close()

    ### 선택된 시간에 따른 평균 입자 면적(mm²) vs. 거리(mm) 그래프 그리기 ###
    plt.figure(figsize=(14, 10))

    for idx, time_value in enumerate(unique_times):
        # 현재 시간에 해당하는 데이터 필터링 (부동 소수점 이슈 방지를 위한 허용 오차 사용)
        time_data = sampled_df[np.isclose(sampled_df['Time'], time_value, atol=1e-2)].copy()

        # 현재 시간에 데이터가 존재하는지 확인
        if time_data.empty:
            print(f"No data found for Time {time_value:.2f}s.")
            continue

        # 거리(mm) 기준으로 빈 정의
        bin_size_px = 50  # 원래 픽셀 단위 빈 크기
        bin_size_mm = bin_size_px * pixels_to_mm  # mm 단위로 변환
        bins_mm = np.arange(0, overall_distance_max + bin_size_mm, bin_size_mm)
        time_data['DistanceBin_mm'] = pd.cut(time_data['Distance to Center (mm)'], bins=bins_mm)

        # 거리 빈별로 평균 거리(mm)와 평균 면적(mm²) 계산
        grouped = time_data.groupby('DistanceBin_mm', observed=False)
        binned_data = grouped.agg({'Distance to Center (mm)': 'mean', 'Particle Area (mm^2)': 'mean'}).reset_index()

        # 빈별 데이터 포인트 수 계산
        binned_data['Count'] = grouped.size().values

        # NaN 값 제거
        binned_data = binned_data.dropna(subset=['Distance to Center (mm)', 'Particle Area (mm^2)'])

        # 거리(mm) 기준으로 정렬
        binned_data = binned_data.sort_values('Distance to Center (mm)')

        # 각 빈에 최소 5개 이상의 데이터 포인트가 있는지 필터링
        min_points_per_bin = 5
        binned_data = binned_data[binned_data['Count'] >= min_points_per_bin]

        # 충분한 데이터가 있는지 확인
        if len(binned_data) >= 1:
            x = binned_data['Distance to Center (mm)'].values
            y = binned_data['Particle Area (mm^2)'].values

            # 평균 값을 선과 마커로 플롯
            plt.plot(x, y, label=f'{time_value:.2f}s', color=colors[idx], linewidth=2, marker='o')
        else:
            print(f"Not enough data points to plot average for Time {time_value:.2f}s.")
            continue

    plt.title(f'Average Particle Area vs. Distance at Selected Times (φ={phi})', fontsize=18)
    plt.xlabel('Distance to Center (mm)', fontsize=14)
    plt.ylabel('Average Particle Area (mm²)', fontsize=14)

    # 범례 설정
    plt.legend(title='Time (s)', fontsize=12, title_fontsize=12,
               loc='upper right', bbox_to_anchor=(1, 1), ncol=1)

    # 축 스케일 팩터 적용
    plt.xlim(overall_distance_min, overall_distance_max * x_scale_factor)
    plt.ylim(overall_area_min, overall_area_max * y_scale_factor)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # 평균 면적 그래프 저장
    average_area_save_path = os.path.join(plots_dir, f'Average_Particle_Area_vs_Distance_Selected_Time_φ={phi}.png')
    plt.savefig(average_area_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved average area plot to: {average_area_save_path}")
    plt.close()

    # 개별 시간 그래프 (평균 추세선 없이) 그리기
    for idx, time_value in enumerate(unique_times):
        # 현재 시간에 해당하는 데이터 필터링 (허용 오차 사용)
        time_data = sampled_df[np.isclose(sampled_df['Time'], time_value, atol=1e-2)].copy()

        # 현재 시간에 데이터가 존재하는지 확인
        if time_data.empty:
            print(f"No data found for Time {time_value:.2f}s.")
            continue

        plt.figure(figsize=(12, 8))

        # 개별 시간에 대한 산점도 그리기
        sns.scatterplot(
            data=time_data,
            x='Distance to Center (mm)',
            y='Particle Area (mm^2)',
            color=colors[idx],
            alpha=0.6,
            edgecolor=None
        )

        # 선택된 데이터에서 현재 시간에 대한 전체 입자 수 계산
        full_time_data = selected_df[np.isclose(selected_df['Time'], time_value, atol=1e-2)]
        particle_count = len(full_time_data)

        # 그래프에 입자 수 표시
        plt.text(0.95, 0.95, f'Particle Count: {particle_count}',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=plt.gca().transAxes,
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.title(f'Particle Area vs. Distance at Time {time_value:.2f}s (φ={phi})', fontsize=18)
        plt.xlabel('Distance to Center (mm)', fontsize=14)
        plt.ylabel('Particle Area (mm²)', fontsize=14)

        # 축 한계 설정
        plt.xlim(overall_distance_min, overall_distance_max * x_scale_factor)
        plt.ylim(overall_area_min, overall_area_max * y_scale_factor)

        # x축 형식 지정
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # 축 눈금을 부동 소수점으로 설정
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=False))
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=False))

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        # 샘플링 비율 주석 추가
        plt.text(
            0.95, 0.90, f'Sampling Rate: {sampling_rate * 100:.0f}%',
            horizontalalignment='right',
            verticalalignment='top',
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )
        # 개별 시간 그래프 저장
        individual_filename = f'Particle_Area_vs_Distance_Time_{time_value:.2f}s_φ={phi}.png'
        individual_save_path = os.path.join(plots_dir, individual_filename)
        plt.savefig(individual_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved individual graph to: {individual_save_path}")
        plt.close()

else:
    print(
        "Centroid positions (X and Y) not found in the data. Please ensure that 'Centroid' measurements are included in ImageJ.")
