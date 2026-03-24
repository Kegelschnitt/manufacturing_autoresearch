class LessonCurator:
    def __init__(self, memory):
        self.memory = memory

    def update_lessons(self, new_lessons):
        for lesson in new_lessons:
            self.memory.add_lesson(lesson["lesson_id"], lesson)
