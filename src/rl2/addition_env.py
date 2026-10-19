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


if __name__ == "__main__":
    AdditionTools.run()
