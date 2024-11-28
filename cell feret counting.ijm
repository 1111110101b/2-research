// 디렉토리 선택 및 파일 리스트 가져오기
dir = getDirectory("디렉토리 선택");
list = getFileList(dir);

// 결과 및 편집된 이미지 저장을 위한 하위 디렉토리 생성
outputDir = dir + "Results/";
File.makeDirectory(outputDir);
imageOutputDir = dir + "ProcessedImages/";
File.makeDirectory(imageOutputDir);

// 배치 모드 활성화 (디스플레이 업데이트 끄기)
setBatchMode(true);

// 요약 결과를 저장할 변수 초기화 (헤더 수정)
summaryData = "Filename,Particle Count,Total Area,Average Area,Average Feret,Average Circularity\n";

// 파일 리스트 순회
for (i = 0; i < list.length; i++) {
    // 이미지 파일인지 확인
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".jpg") || endsWith(list[i], ".jpeg") || endsWith(list[i], ".png")) {
        // 파일 이름에서 숫자 추출 (마지막 점 이전)
        fileName = substring(list[i], 0, lastIndexOf(list[i], "."));
        num = parseInt(fileName);

        // 숫자가 5의 배수인지 확인
        if (num % 5 == 0) {
            // 이미지 열기
            open(dir + list[i]);

            // 선택 영역 내부 지우기
            makePolygon(1680, 0, 3840, 2160, 3840, 0); // 폴리곤 좌표 확인
            run("Clear"); // 선택 영역 내부 지우기
            run("Select None"); // 선택 해제

            // 1. 8비트 그레이스케일로 변환
            run("8-bit");

            // 2. 배경 제거 (Rolling Ball Half Width = 50)
            run("Subtract Background...", "rolling=50 light background");

            // 3. 이미지 반전
            run("Invert");

            // 4. 임계값 설정 (자동 임계값 설정)
            // setAutoThreshold("Default");
            setThreshold(6, 255); // 수동 임계값 설정 (필요 시 주석 해제하고 값 조정)

            // 임계값 적용 후 이진화
            run("Convert to Mask");

            // 결과 테이블을 초기화하여 이전 결과가 남지 않도록 함
            run("Clear Results");

            // 측정 항목 설정 (Area, Feret's Diameter, Circularity 포함)
            run("Set Measurements...", "area centroid shape feret's display redirect=None decimal=3");

            // Analyze Particles 실행 (필요에 따라 크기 및 원형도 조정)
            run("Analyze Particles...", "size=150-Infinity circularity=0.1-1.00 clear overlay");

            // 결과 테이블을 CSV 파일로 저장
            saveAs("Results", outputDir + fileName + "_results.csv");

            // 현재 이미지의 입자 수 및 총합 계산
            particleCount = nResults;
            totalArea = 0;
            totalFeret = 0;
            totalCircularity = 0;

            for (j = 0; j < particleCount; j++) {
                totalArea += getResult("Area", j);
                totalFeret += getResult("Feret", j);
                totalCircularity += getResult("Circ.", j);
            }

            // 평균 계산
            if (particleCount > 0) {
                avgArea = totalArea / particleCount;
                avgFeret = totalFeret / particleCount;
                avgCircularity = totalCircularity / particleCount;
            } else {
                avgArea = 0;
                avgFeret = 0;
                avgCircularity = 0;
            }

            // 요약 데이터를 문자열에 추가
            summaryData += fileName + "," + particleCount + "," + totalArea + "," + avgArea + "," + avgFeret + "," + avgCircularity + "\n";

            // 결과 테이블 초기화
            run("Clear Results");

            // 처리된 이미지를 JPG로 저장
            saveAs("Jpeg", imageOutputDir + fileName + "_processed.jpg");

            // 이미지 닫기
            close();
        }
    }
}

// 배치 모드 비활성화
setBatchMode(false);

// 요약 데이터를 파일로 저장
summaryFile = outputDir + "Summary.csv";
File.saveString(summaryData, summaryFile);

// 요약 파일을 열어 화면에 표시
open(summaryFile);
