#!/usr/bin/python3
import os
import progressbar


class ProgressBar(object):
    """
        Function: it show the progress.
    """
    def __init__(self, text='unknown', maxval=0):
        self.bar = progressbar.ProgressBar(term_width=100, maxval=maxval, widgets=[
            progressbar.Bar('=', '[', ']'), ' ',
            progressbar.Percentage(), ' ',
            progressbar.Timer(), ' ',
            progressbar.ETA()])
        self.step = 2047
        if maxval < 2 ** 10 * 20:
            self.step = 127
        if maxval < 2 ** 10:
            self.step = 1
        print('[\033[32m', text, '\033[0m:', maxval, ']')
        self.bar.start()

    def update(self, progress):
        if progress & self.step == 0:
            self.bar.update(progress)

    def finish(self):
        self.bar.finish()


class DirectoryWriter(dict):
    """
        function: it is a writer to write buffer content into a file
    """
    def __init__(self, directory, filename):
        super().__init__()
        self.directory = directory
        self.filename = filename
        self.openings = 0

    def __getitem__(self, key):
        if self.openings > 128:
            self.close()
        if key not in self or dict.__getitem__(self, key) is None:
            keydir = os.path.join(self.directory, key)
            keyfile = os.path.join(keydir, self.filename)
            os.makedirs(keydir, exist_ok=True)
            dict.__setitem__(self, key, open(keyfile, 'w' if key not in self else 'a'))
            self.openings += 1
        return dict.__getitem__(self, key)

    def write(self, key, content):
        self.__getitem__(key).write(content)

    def close(self):
        for key in self:
            if dict.__getitem__(self, key) is not None:
                dict.__getitem__(self, key).close()
                dict.__setitem__(self, key, None)
        self.openings = 0
