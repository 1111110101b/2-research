// ���丮 ���� �� ���� ����Ʈ ��������
dir = getDirectory("���丮 ����");
list = getFileList(dir);

// ��� �� ������ �̹��� ������ ���� ���� ���丮 ����
outputDir = dir + "Results/";
File.makeDirectory(outputDir);
imageOutputDir = dir + "ProcessedImages/";
File.makeDirectory(imageOutputDir);

// ��ġ ��� Ȱ��ȭ (���÷��� ������Ʈ ����)
setBatchMode(true);

// ��� ����� ������ ���� �ʱ�ȭ (��� ����)
summaryData = "Filename,Particle Count,Total Area,Average Area,Average Feret,Average Circularity\n";

// ���� ����Ʈ ��ȸ
for (i = 0; i < list.length; i++) {
    // �̹��� �������� Ȯ��
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".jpg") || endsWith(list[i], ".jpeg") || endsWith(list[i], ".png")) {
        // ���� �̸����� ���� ���� (������ �� ����)
        fileName = substring(list[i], 0, lastIndexOf(list[i], "."));
        num = parseInt(fileName);

        // ���ڰ� 5�� ������� Ȯ��
        if (num % 5 == 0) {
            // �̹��� ����
            open(dir + list[i]);

            // ���� ���� ���� �����
            makePolygon(1680, 0, 3840, 2160, 3840, 0); // ������ ��ǥ Ȯ��
            run("Clear"); // ���� ���� ���� �����
            run("Select None"); // ���� ����

            // 1. 8��Ʈ �׷��̽����Ϸ� ��ȯ
            run("8-bit");

            // 2. ��� ���� (Rolling Ball Half Width = 50)
            run("Subtract Background...", "rolling=50 light background");

            // 3. �̹��� ����
            run("Invert");

            // 4. �Ӱ谪 ���� (�ڵ� �Ӱ谪 ����)
            // setAutoThreshold("Default");
            setThreshold(6, 255); // ���� �Ӱ谪 ���� (�ʿ� �� �ּ� �����ϰ� �� ����)

            // �Ӱ谪 ���� �� ����ȭ
            run("Convert to Mask");

            // ��� ���̺��� �ʱ�ȭ�Ͽ� ���� ����� ���� �ʵ��� ��
            run("Clear Results");

            // ���� �׸� ���� (Area, Feret's Diameter, Circularity ����)
            run("Set Measurements...", "area centroid shape feret's display redirect=None decimal=3");

            // Analyze Particles ���� (�ʿ信 ���� ũ�� �� ������ ����)
            run("Analyze Particles...", "size=150-Infinity circularity=0.1-1.00 clear overlay");

            // ��� ���̺��� CSV ���Ϸ� ����
            saveAs("Results", outputDir + fileName + "_results.csv");

            // ���� �̹����� ���� �� �� ���� ���
            particleCount = nResults;
            totalArea = 0;
            totalFeret = 0;
            totalCircularity = 0;

            for (j = 0; j < particleCount; j++) {
                totalArea += getResult("Area", j);
                totalFeret += getResult("Feret", j);
                totalCircularity += getResult("Circ.", j);
            }

            // ��� ���
            if (particleCount > 0) {
                avgArea = totalArea / particleCount;
                avgFeret = totalFeret / particleCount;
                avgCircularity = totalCircularity / particleCount;
            } else {
                avgArea = 0;
                avgFeret = 0;
                avgCircularity = 0;
            }

            // ��� �����͸� ���ڿ��� �߰�
            summaryData += fileName + "," + particleCount + "," + totalArea + "," + avgArea + "," + avgFeret + "," + avgCircularity + "\n";

            // ��� ���̺� �ʱ�ȭ
            run("Clear Results");

            // ó���� �̹����� JPG�� ����
            saveAs("Jpeg", imageOutputDir + fileName + "_processed.jpg");

            // �̹��� �ݱ�
            close();
        }
    }
}

// ��ġ ��� ��Ȱ��ȭ
setBatchMode(false);

// ��� �����͸� ���Ϸ� ����
summaryFile = outputDir + "Summary.csv";
File.saveString(summaryData, summaryFile);

// ��� ������ ���� ȭ�鿡 ǥ��
open(summaryFile);
