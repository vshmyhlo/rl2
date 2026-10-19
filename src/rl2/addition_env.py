import verifiers.v1 as vf


class AdditionData(vf.TaskData):
    answer: int


class AdditionTools(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = None

    @vf.tool
    async def add(self, a: int, b: int) -> int:
        """Add two integers."""
        return a + b


class AdditionTask(vf.Task[AdditionData]):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [AdditionTools(config=vf.ToolsetConfig())]

    @vf.reward
    async def correct_answer(self, trace: vf.Trace) -> float:
        return float(trace.last_reply.strip() == str(self.data.answer))


class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    def load(self) -> list[AdditionTask]:
        return [
            AdditionTask(
                data=AdditionData(
                    id=str(index),
                    prompt=f"What is {a} + {b}?",
                    answer=a + b,
                    system_prompt="Use the add tool, then reply with only the resulting integer.",
                ),
                config=self.config.task,
            )
            for index, (a, b) in enumerate([(17, 25), (8, 13), (0, 9), (-4, 7), (31, 12)])
        ]


if __name__ == "__main__":
    AdditionTools.run()
