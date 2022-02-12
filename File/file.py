import hashlib
import os
import re


class File:
    def mergedFile(self, path, fileName, files: list):
        filesExist = self.getFiles(path)
        if fileName in filesExist:
            return True

        fileList = ["" for _ in files]
        for file in files:
            if file not in filesExist:
                return False
            regexStr = '(?<=' + fileName.replace(".", "\\.") + '\\.)[0-9]+'
            index = re.search(regexStr, file)
            if index is None:
                return False
            index = int(index.group(0))
            fileList[index] = os.path.join(path, file)

        with open(os.path.join(path, fileName), 'wb') as f:
            for file in fileList:
                f.write(self.binaryRead(file))
        return True

    def splitFile(self, path, fileName, size):
        filesExist = self.getFiles(path)
        if fileName not in filesExist:
            return False

        with open(os.path.join(path, fileName), 'rb') as f:
            data = f.read()
            dataList = [data[i:i + size if i + size < len(data) else len(data)] for i in range(0, len(data), size)]

        for i in range(len(dataList)):
            self.binaryWrite(os.path.join(path, f"{fileName}.{str(i)}"), dataList[i])

        return True

    def md5(self, fileName):
        data = self.binaryRead(fileName)
        file_md5 = hashlib.md5(data).hexdigest()
        return file_md5

    @staticmethod
    def binaryRead(fileName):
        with open(fileName, 'rb') as f:
            return f.read()

    @staticmethod
    def binaryWrite(fileName, data):
        with open(fileName, 'wb') as f:
            f.write(data)

    @staticmethod
    def getFiles(path):
        files = []
        for filename in os.listdir(path):
            if not os.path.isdir(os.path.join(path, filename)):
                files.append(filename)

        return files
