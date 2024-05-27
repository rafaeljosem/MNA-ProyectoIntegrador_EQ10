import os


class Logger:

    def open_or_create_log(self, path='/', filename='generator.log'):
        full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

        return open(full_path, 'a', encoding='utf8')

    def write_to_log(self, path, filename, line: str):
        f = self.open_or_create_log(path, filename)
        f.write(line)
        f.close()
