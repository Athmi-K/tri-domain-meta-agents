from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from unittest.mock import MagicMock
from types import SimpleNamespace

from agents import career_agent
from langchain_agents.meta_lc_agent import _build_chat_explainability
from langchain_agents import career_lc_agent
from schemas.chat import ChatResponse
from tools import career_tools
from core.intent_detector import detect_intent


class CareerPromptTests(TestCase):
    def _request(self, query: str) -> SimpleNamespace:
        return SimpleNamespace(
            query=query,
            current_skills=["Python"],
            target_role="Software Engineer",
            location="Bangalore",
            experience_level="entry",
            years_experience=0,
            current_level="beginner",
            timeline_months=6,
            resume_text="",
        )

    def test_prompt_prioritizes_current_query(self):
        query = "Explain method overloading and overriding in Java."

        prompt = career_agent._build_summary_prompt(
            current_query=query,
            target_role="Software Engineer",
            current_skills=["Python"],
            gap_data={},
            jobs_data={},
            salary_data={},
            path_data={},
            resume_data=None,
        )

        self.assertIn(query, prompt)
        self.assertIn("unrelated career recommendations or force a career roadmap", prompt.replace("\n", " "))
        self.assertIn("For technical questions", prompt)

    @patch.object(career_agent, "skill_gap_analyzer", return_value={"missing_skills": ["Java"]})
    @patch.object(career_agent, "job_search", return_value={"jobs": []})
    @patch.object(career_agent, "salary_benchmark", return_value={})
    @patch.object(career_agent, "learning_path_generator", return_value={})
    @patch.object(career_agent, "call_llm", return_value={"summary": "Java explanation"})
    def test_run_passes_query_and_preserves_response_shape(
        self,
        call_llm,
        learning_path_generator,
        salary_benchmark,
        job_search,
        skill_gap_analyzer,
    ):
        query = "Explain method overloading and overriding in Java."

        result = career_agent.run(self._request(query))

        prompt = call_llm.call_args.args[1]
        self.assertIn(query, prompt)
        self.assertEqual(result["summary"], "Java explanation")
        self.assertTrue(
            {
                "target_role",
                "location",
                "experience_level",
                "current_skills",
                "skill_gap",
                "jobs",
                "salary",
                "learning_path",
                "resume_analysis",
                "summary",
                "tools_used",
                "tool_outputs",
                "sources",
            }.issubset(result)
        )

    @patch.object(career_agent, "skill_gap_analyzer")
    @patch.object(career_agent, "job_search")
    @patch.object(career_agent, "salary_benchmark")
    @patch.object(career_agent, "learning_path_generator")
    @patch.object(career_agent, "call_llm", return_value={"summary": "Python explanation"})
    def test_technical_question_does_not_run_career_tools(
        self,
        call_llm,
        learning_path_generator,
        salary_benchmark,
        job_search,
        skill_gap_analyzer,
    ):
        result = career_agent.run(self._request("What is a Python interface?"))

        skill_gap_analyzer.assert_not_called()
        job_search.assert_not_called()
        salary_benchmark.assert_not_called()
        learning_path_generator.assert_not_called()
        self.assertEqual(result["summary"], "Python explanation")

    def test_java_python_and_dbms_questions_return_relevant_mocked_answers(self):
        queries_and_answers = (
            ("Explain Java interfaces.", "Java interfaces define contracts."),
            ("What is a Python decorator?", "A Python decorator wraps a function."),
            ("What is a DBMS?", "A DBMS stores and manages structured data."),
        )

        with patch.object(
            career_agent,
            "call_llm",
            side_effect=[{"summary": answer} for _, answer in queries_and_answers],
        ):
            with patch.object(career_agent, "skill_gap_analyzer") as skill_gap:
                with patch.object(career_agent, "job_search") as job_search:
                    with patch.object(career_agent, "salary_benchmark") as salary:
                        with patch.object(
                            career_agent, "learning_path_generator"
                        ) as learning_path:
                            for query, expected in queries_and_answers:
                                with self.subTest(query=query):
                                    result = career_agent.run(self._request(query))
                                    self.assertIn(expected, result["summary"])

        skill_gap.assert_not_called()
        job_search.assert_not_called()
        salary.assert_not_called()
        learning_path.assert_not_called()

    @patch.object(career_agent, "skill_gap_analyzer", return_value={"missing_skills": []})
    @patch.object(career_agent, "job_search", return_value={"jobs": []})
    @patch.object(career_agent, "salary_benchmark", return_value={})
    @patch.object(
        career_agent,
        "learning_path_generator",
        return_value={"timeline_months": 3, "phases": []},
    )
    @patch.object(career_agent, "call_llm", return_value={"summary": "Placement plan"})
    def test_placement_preparation_is_query_aware(
        self,
        call_llm,
        learning_path_generator,
        salary_benchmark,
        job_search,
        skill_gap_analyzer,
    ):
        query = "Create a software placement preparation plan."
        result = career_agent.run(self._request(query))

        self.assertEqual(result["summary"], "Placement plan")
        self.assertIn(query, call_llm.call_args.args[1])
        learning_path_generator.assert_called_once()
        self.assertEqual(
            learning_path_generator.call_args.kwargs["timeline_months"],
            6,
        )

    def test_exact_placement_query_preserves_four_weeks_daily_time_and_language_neutrality(self):
        query = (
            "I have only 1 hour per day to prepare for software engineering "
            "placements. Create a realistic 4-week schedule with daily tasks."
        )
        prompt = career_agent._build_summary_prompt(
            current_query=query,
            target_role="Software Engineer",
            current_skills=[],
            gap_data={},
            jobs_data={},
            salary_data={},
            path_data={},
            resume_data=None,
        )

        self.assertIn("exactly 4 weeks (28 daily entries)", prompt)
        self.assertIn("approximately 1 hour(s) per day", prompt)
        self.assertIn("Do not choose or force a programming language", prompt)

    def test_exact_placement_query_routes_to_career(self):
        result = detect_intent(
            "I have only 1 hour per day to prepare for software engineering "
            "placements. Create a realistic 4-week schedule with daily tasks."
        )

        self.assertEqual(result["domains"], ["career"])

    def test_explicit_four_week_timeline_overrides_profile_timeline(self):
        self.assertEqual(
            career_agent._requested_timeline_months(
                "Create a realistic 4-week placement schedule."
            ),
            1,
        )

    def test_unrelated_query_does_not_route_to_career(self):
        result = detect_intent("Write a poem about the ocean.")

        self.assertEqual(result["domains"], ["general"])

    def test_specified_language_is_preserved_in_schedule_constraints(self):
        prompt = career_agent._build_summary_prompt(
            current_query="Create a 4-week Python placement schedule.",
            target_role="Software Engineer",
            current_skills=[],
            gap_data={},
            jobs_data={},
            salary_data={},
            path_data={},
            resume_data=None,
        )

        self.assertIn("Use the requested language: python", prompt)

    @patch.object(career_agent, "skill_gap_analyzer", return_value={})
    @patch.object(career_agent, "job_search", return_value={"jobs": []})
    @patch.object(career_agent, "salary_benchmark", return_value={})
    @patch.object(career_agent, "learning_path_generator", return_value={})
    @patch.object(career_agent, "call_llm", return_value={"summary": "Job search"})
    def test_invalid_location_is_not_forwarded_as_profile_fact(
        self,
        call_llm,
        learning_path_generator,
        salary_benchmark,
        job_search,
        skill_gap_analyzer,
    ):
        request = self._request("Find jobs for my target role.")
        request.location = "mani"

        career_agent.run(request)

        self.assertEqual(job_search.call_args.kwargs["location"], "")

    def test_explainability_lists_only_tools_that_executed(self):
        request = self._request("Explain Java interfaces.")
        explanation = _build_chat_explainability(
            "career",
            request,
            {
                "tools_used": [],
                "confidence": None,
            },
        )

        self.assertEqual(explanation["tools_invoked"], [])
        self.assertEqual(explanation["retrieved_sources"], [])
        self.assertIn("No retrieved sources", explanation["limitations"][0])

    def test_explainability_reports_actual_tool_status_and_source(self):
        request = self._request("Find software engineering jobs.")
        explanation = _build_chat_explainability(
            "career",
            request,
            {
                "agent_type": "langchain",
                "tools_used": ["job_search_tool", "salary_benchmark_tool"],
                "tool_outputs": {
                    "job_search_tool": {"status": "completed"},
                    "salary_benchmark_tool": {"status": "failed"},
                },
                "sources": ["career_kb.txt"],
                "confidence": 1.5,
            },
        )

        self.assertEqual(explanation["original_query"], request.query)
        self.assertEqual(explanation["selected_agent"], "langchain")
        self.assertEqual(
            explanation["tool_statuses"]["salary_benchmark_tool"], "failed"
        )
        self.assertEqual(explanation["retrieved_sources"], ["career_kb.txt"])
        self.assertIn("salary_benchmark_tool", explanation["limitations"][0])
        self.assertIn("No confidence score", explanation["confidence_explanation"])

    def test_legacy_fallback_explainability_reports_tools_that_ran(self):
        request = self._request(
            "I have only 1 hour per day to prepare for software engineering "
            "placements. Create a realistic 4-week schedule with daily tasks."
        )
        explanation = _build_chat_explainability(
            "career",
            request,
            {
                "agent_type": "fallback",
                "tools_used": ["learning_path_generator"],
                "tool_outputs": {
                    "learning_path_generator": {
                        "status": "completed",
                        "output": {"timeline_months": 1},
                    }
                },
                "sources": [],
            },
        )

        self.assertEqual(explanation["selected_agent"], "fallback")
        self.assertEqual(
            explanation["tool_statuses"]["learning_path_generator"],
            "completed",
        )
        self.assertIn(
            "Target role",
            explanation["profile_values_used"],
        )

    @patch.object(career_lc_agent, "career_agent")
    def test_langchain_path_preserves_query_and_api_tool_output_shape(self, agent):
        message = MagicMock()
        message.content = '{"recommendation": "Explain Java interfaces."}'
        message.type = "ai"
        agent.invoke.return_value = {"messages": [message]}

        result = career_lc_agent.run(self._request("Explain Java interfaces."))

        user_input = agent.invoke.call_args.args[0]["messages"][0]["content"]
        self.assertIn("Explain Java interfaces.", user_input)
        self.assertNotIn("data scientist", user_input)
        self.assertIsInstance(result["tool_outputs"], dict)
        self.assertNotIn("confidence", result)
        self.assertEqual(result["sources"], [])

    @patch.object(career_lc_agent, "career_agent")
    def test_langchain_tool_execution_records_success_failure_and_sources(self, agent):
        final_message = SimpleNamespace(
            type="ai",
            content='{"recommendation": "Job result"}',
        )
        success = SimpleNamespace(
            type="tool",
            name="job_search_tool",
            status="completed",
            content={"jobs": [], "sources": ["jobs-api"]},
        )
        failure = SimpleNamespace(
            type="tool",
            name="salary_benchmark_tool",
            status="failed",
            content={"error": "service unavailable"},
        )
        agent.invoke.return_value = {
            "messages": [success, failure, final_message]
        }

        result = career_lc_agent.run(self._request("Find jobs and salary data."))

        self.assertEqual(result["sources"], ["jobs-api"])
        self.assertEqual(result["tool_outputs"]["job_search_tool"]["status"], "completed")
        self.assertEqual(result["tool_outputs"]["salary_benchmark_tool"]["status"], "failed")

    @patch.object(career_lc_agent, "career_agent")
    def test_exact_placement_query_direct_llm_has_execution_accurate_explainability(
        self,
        agent,
    ):
        message = MagicMock()
        message.type = "ai"
        message.content = (
            '{"recommendation": "A four-week Java placement schedule.", '
            '"reason": "The query requests a daily schedule."}'
        )
        agent.invoke.return_value = {"messages": [message]}
        request = self._request(
            "I have only 1 hour per day to prepare for software engineering "
            "placements. Create a realistic 4-week schedule with daily tasks. "
            "I am learning Java."
        )

        with patch.object(
            career_lc_agent,
            "_retrieve_career_context",
            return_value={"status": "empty", "context": "", "sources": []},
        ):
            result = career_lc_agent.run(request)
        explanation = result["explainability"]

        self.assertEqual(explanation["selected_agent"], "langchain")
        self.assertEqual(explanation["domain"], "career")
        self.assertEqual(explanation["original_query"], request.query)
        self.assertEqual(explanation["profile_values_used"], {})
        self.assertEqual(explanation["tools_invoked"], [])
        self.assertEqual(explanation["tool_statuses"], {})
        self.assertEqual(explanation["retrieved_sources"], [])
        self.assertEqual(explanation["retrieval_status"], "empty")
        self.assertIn("directly by the LangChain Career LLM", explanation["decision_factors"][0])
        self.assertIn("No Career tools", explanation["limitations"][0])
        self.assertIn("No confidence score", explanation["confidence_explanation"])
        self.assertIn("Requested duration: 4 weeks", explanation["data_used"])
        self.assertIn("Daily study limit: 1 hour per day", explanation["data_used"])
        self.assertIn("Requested language: java", explanation["data_used"])
        self.assertIn("missing_profile_fields", explanation)
        self.assertIn("next_steps", explanation)
        self.assertIn("processing_steps", explanation)

    @patch.object(career_lc_agent, "career_agent")
    def test_career_explainability_records_tool_profile_status_and_sources(self, agent):
        tool_message = SimpleNamespace(
            type="tool",
            name="learning_path_tool",
            status="completed",
            content={"timeline_months": 1, "source": "career_kb.txt"},
        )
        final_message = SimpleNamespace(
            type="ai",
            content='{"recommendation": "Schedule", "reason": "Tool result"}',
        )
        agent.invoke.return_value = {"messages": [tool_message, final_message]}
        request = self._request("Create a one-month placement plan.")

        result = career_lc_agent.run(request)
        explanation = result["explainability"]

        self.assertEqual(explanation["tools_invoked"], ["learning_path_tool"])
        self.assertEqual(
            explanation["tool_statuses"]["learning_path_tool"],
            "completed",
        )
        self.assertEqual(explanation["retrieved_sources"], ["career_kb.txt"])
        self.assertEqual(explanation["profile_values_used"]["Target role"], "Software Engineer")

    @patch.object(career_lc_agent, "career_agent")
    def test_career_explainability_records_failed_tool_without_sources(self, agent):
        tool_message = SimpleNamespace(
            type="tool",
            name="learning_path_tool",
            status="failed",
            content={"error": "unavailable"},
        )
        final_message = SimpleNamespace(
            type="ai",
            content='{"recommendation": "Limited schedule", "reason": "Tool failed"}',
        )
        agent.invoke.return_value = {"messages": [tool_message, final_message]}

        with patch.object(
            career_lc_agent,
            "_retrieve_career_context",
            return_value={"status": "empty", "context": "", "sources": []},
        ):
            result = career_lc_agent.run(self._request("Create a learning plan."))
        explanation = result["explainability"]

        self.assertEqual(explanation["tool_statuses"]["learning_path_tool"], "failed")
        self.assertEqual(explanation["retrieved_sources"], [])
        self.assertTrue(any("failed" in item for item in explanation["limitations"]))

    @patch.object(career_lc_agent, "career_agent")
    @patch.object(career_lc_agent, "_retrieve_career_context")
    def test_exact_placement_query_passes_retrieved_context_and_sources(
        self,
        retrieve_context,
        agent,
    ):
        retrieve_context.return_value = {
            "status": "success",
            "context": (
                "\n\nRELEVANT CAREER KNOWLEDGE:\n"
                "[Source: career_kb.txt]\nDSA interview preparation topics."
            ),
            "sources": ["career_kb.txt"],
        }
        final_message = SimpleNamespace(
            type="ai",
            content="Use the retrieved plan.",
        )
        agent.invoke.return_value = {"messages": [final_message]}
        request = self._request(
            "I am a third-year Computer Science engineering student preparing "
            "for software engineering placements. I know basic Java, arrays, "
            "strings, and loops, but I struggle to solve coding problems "
            "independently. Help me prepare for campus placements over the "
            "next 30 days with one hour of study per day. Identify the "
            "important Java and DSA topics I should focus on, suggest a daily "
            "practice plan, explain how to improve problem-solving skills, and "
            "recommend how I should prepare for aptitude tests and technical "
            "interviews based on my current skill level."
        )

        result = career_lc_agent.run(request)
        prompt = agent.invoke.call_args.args[0]["messages"][0]["content"]

        retrieve_context.assert_called_once_with(request.query)
        self.assertIn("DSA interview preparation topics.", prompt)
        self.assertEqual(result["sources"], ["career_kb.txt"])
        self.assertNotIn(
            "no structured execution metadata was returned",
            result["reason"].casefold(),
        )
        self.assertIn("retrieved Career documents", result["reason"])
        self.assertEqual(
            result["explainability"]["retrieval_status"],
            "success",
        )
        self.assertEqual(
            result["explainability"]["retrieved_sources"],
            ["career_kb.txt"],
        )
        self.assertIn("Requested duration: 30 days", result["explainability"]["data_used"])
        self.assertIn("Daily study limit: 1 hour per day", result["explainability"]["data_used"])
        self.assertIn("Requested language: java", result["explainability"]["data_used"])
        self.assertNotIn("directly by the LangChain Career LLM",
                         result["explainability"]["decision_factors"][0])

    @patch.object(career_lc_agent, "career_agent")
    @patch.object(career_lc_agent, "_retrieve_career_context")
    def test_career_retrieval_empty_result_is_reported(self, retrieve_context, agent):
        retrieve_context.return_value = {
            "status": "empty",
            "context": "",
            "sources": [],
        }
        agent.invoke.return_value = {
            "messages": [
                SimpleNamespace(type="ai", content='{"recommendation": "Fallback"}')
            ]
        }

        result = career_lc_agent.run(
            self._request("Create a software placement preparation plan.")
        )

        self.assertEqual(result["explainability"]["retrieval_status"], "empty")
        self.assertIn(
            "returned no documents",
            " ".join(result["explainability"]["limitations"]),
        )
        self.assertEqual(result["sources"], [])

    @patch.object(career_lc_agent, "career_agent")
    @patch.object(career_lc_agent, "_retrieve_career_context")
    def test_career_retrieval_failure_is_reported_without_sources(
        self,
        retrieve_context,
        agent,
    ):
        retrieve_context.return_value = {
            "status": "failed",
            "context": "",
            "sources": [],
            "error": "index unavailable",
        }
        agent.invoke.return_value = {
            "messages": [
                SimpleNamespace(type="ai", content='{"recommendation": "Fallback"}')
            ]
        }

        result = career_lc_agent.run(
            self._request("Create a software placement preparation plan.")
        )

        explanation = result["explainability"]
        self.assertEqual(explanation["retrieval_status"], "failed")
        self.assertIn("index unavailable", " ".join(explanation["limitations"]))
        self.assertEqual(explanation["retrieved_sources"], [])

    def test_career_explainability_reports_missing_profile_fields(self):
        request = SimpleNamespace(
            query="Create a learning plan for software engineering placements.",
            current_skills=[],
            target_role=None,
            location=None,
            experience_level=None,
            timeline_months=None,
        )

        explanation = career_lc_agent._build_career_explainability(
            request,
            {"agent_type": "langchain", "tools_used": [], "tool_outputs": []},
        )

        self.assertEqual(
            explanation["missing_profile_fields"],
            ["Target role", "Current skills", "Experience level"],
        )

    def test_partial_career_explainability_is_completed_without_overwriting_execution(self):
        request = self._request("Find software engineering jobs.")
        result = {
            "agent_type": "langchain",
            "tools_used": ["job_search_tool"],
            "tool_outputs": {
                "job_search_tool": {"status": "completed"},
            },
            "sources": ["jobs-api"],
            "explainability": {
                "tools_invoked": ["job_search_tool"],
                "retrieved_sources": ["jobs-api"],
            },
        }

        from langchain_agents import meta_lc_agent
        with patch.object(meta_lc_agent, "career_run", return_value=result), \
                patch.object(meta_lc_agent, "_build_chat_request", return_value=request):
            completed = meta_lc_agent.run_chat_agent(
                db=MagicMock(),
                user_id="user-1",
                domain="career",
                query=request.query,
            )

        self.assertEqual(
            completed["explainability"]["tools_invoked"],
            ["job_search_tool"],
        )
        self.assertEqual(
            completed["explainability"]["retrieved_sources"],
            ["jobs-api"],
        )
        self.assertIn("missing_profile_fields", completed["explainability"])

    @patch.object(career_tools, "_call_llm", return_value={
        "phases": [
            {"phase": 1, "duration_weeks": 4},
            {"phase": 2, "duration_weeks": 8},
        ]
    })
    @patch("rag.retriever.retrieve_as_context", return_value="")
    def test_learning_path_durations_and_weekly_schedule_are_consistent(
        self,
        retrieve_as_context,
        llm_call,
    ):
        result = career_tools.learning_path_generator(
            target_role="Data Analyst",
            current_level="beginner",
            timeline_months=3,
            missing_skills=["SQL"],
        )

        self.assertEqual(result["total_weeks_required"], 12)
        self.assertEqual(result["weekly_hours_needed"], 10.0)
        self.assertTrue(result["feasible"])
        self.assertEqual(
            sum(phase["duration_weeks"] for phase in result["phases"]),
            result["total_weeks_required"],
        )
        self.assertEqual(result["timeline_months"], 3)

    def test_learning_path_prompt_uses_requested_timeline(self):
        prompt = career_agent._build_summary_prompt(
            current_query="Create a 3-month learning plan for a data analyst.",
            target_role="Data Analyst",
            current_skills=["Python"],
            gap_data={"missing_skills": ["SQL"]},
            jobs_data={},
            salary_data={},
            path_data={"timeline_months": 3, "total_weeks_required": 12},
            resume_data=None,
        )

        self.assertIn("timeline_months': 3", prompt)
        self.assertIn("3-month learning plan", prompt)

    def test_resume_prompt_does_not_add_unverified_achievements(self):
        prompt = career_agent._format_resume_result(
            {
                "semantic_match_score": 65,
                "required_skills": ["Python"],
                "skills_not_evidenced": ["Docker"],
                "suggestions": ["Add evidence only when genuinely present."],
            }
        )

        self.assertNotIn("LeetCode", prompt)
        self.assertNotIn("certification", prompt.casefold())
        self.assertIn("only when genuinely present", prompt)

    def test_resume_advice_keeps_unsupported_achievements_conditional(self):
        resume_data = {
            "semantic_match_score": 65,
            "required_skills": ["Python"],
            "skills_not_evidenced": ["Docker"],
            "suggestions": [
                "If you have completed a certification, add it with evidence; "
                "otherwise do not claim it."
            ],
        }

        formatted = career_agent._format_resume_result(resume_data)

        self.assertIn("If you have completed", formatted)
        self.assertNotIn("You completed", formatted)

    def test_chat_response_serializes_career_execution_metadata(self):
        response = ChatResponse(
            conversation_id="conversation-1",
            domain="career",
            answer="Job result",
            tools_used=["job_search_tool"],
            tool_outputs={
                "job_search_tool": {"status": "completed", "output": {"jobs": []}}
            },
            explainability={
                "original_query": "Find jobs.",
                "selected_agent": "langchain",
                "tools_invoked": ["job_search_tool"],
            },
        )

        payload = response.model_dump()

        self.assertEqual(payload["tools_used"], ["job_search_tool"])
        self.assertEqual(payload["tool_outputs"]["job_search_tool"]["status"], "completed")
        self.assertEqual(payload["explainability"]["selected_agent"], "langchain")


if __name__ == "__main__":
    import unittest

    unittest.main()