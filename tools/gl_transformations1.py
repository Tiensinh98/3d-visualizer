import re
import sqlite3


connection = sqlite3.connect(":memory:")
connection.create_function('regexp', 2, lambda pattern, string: True if len(re.findall(pattern, string)) else False)
cursor = connection.cursor()
cursor.executescript("""
    CREATE TABLE IF NOT EXISTS
    `TestTable` (
        `test_id` INTEGER PRIMARY KEY NOT NULL,
        `test_column` TEXT NOT NULL);
    CREATE UNIQUE INDEX IF NOT EXISTS `TestColumnIndex` ON `TestTable`(`test_column`);
""")
cursor.execute("INSERT INTO `TestTable` (`test_id`, `test_column`) VALUES (?, ?)", (1, 'Sinh'))
cursor.execute("INSERT INTO `TestTable` (`test_id`, `test_column`) VALUES (?, ?)", (2, 'Sinh1'))
cursor.execute("INSERT INTO `TestTable` (`test_id`, `test_column`) VALUES (?, ?)", (3, 'Huong'))
cursor.execute("SELECT `test_column` FROM `TestTable` WHERE regexp(?, `test_column`)", ("^S", ))
print(cursor.fetchall())
