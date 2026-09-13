import { BookOpen, DollarSign, TrendingUp } from "lucide-react";
import { useMemo, useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { MetricCard } from "@/components/common/MetricCard";
import { StatCard } from "@/components/common/StatCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useProfile, useQueryMutation } from "@/hooks";
import { getErrorMessage } from "@/services";
import { formatCurrency } from "@/utils";
import { buildCareerPageData } from "@/utils/profileInsights";
import type { QueryRequest, QueryResponse } from "@/types";

export function CareerPage() {
  const { data: profile } = useProfile();
  const queryMutation = useQueryMutation();
  const [result, setResult] = useState<QueryResponse | null>(null);

  const careerData = useMemo(() => buildCareerPageData(profile), [profile]);
  const { skills, currentSalary } = careerData;
  const targetRoleLabel = profile?.career?.target_role || "Set target role";
  const targetRoleChange = profile?.career?.target_role
    ? `Progress toward ${profile.career.target_role}`
    : "Add a target role for tailored recommendations";
  const resumeTip = profile?.career?.resume
    ? `Update resume with ${profile?.career?.target_role || "career"} achievements`
    : "Add your resume summary to improve guidance";

  const buildRequest = (): QueryRequest => ({
    name: "User",
    age: profile?.general?.age || 25,
    query: `Career guidance for ${profile?.career?.target_role || "my target role"}`,
    domain: "career",
    current_skills: profile?.career?.current_skills || [],
    target_role: profile?.career?.target_role || "",
    experience_level: profile?.career?.experience_level || "",
    location: profile?.general?.location || "Bangalore",
    years_experience: 0,
    current_level: profile?.career?.experience_level || "beginner",
    timeline_months: 6,
    resume_text: profile?.career?.resume_text || profile?.career?.resume || "",
  });

  const handleRunCareerAgent = async () => {
    try {
      const res = await queryMutation.mutateAsync(buildRequest());
      setResult(res);
    } catch (err) {
      console.error(err);
      setResult({
        status: "error",
        message: getErrorMessage(err),
      } as QueryResponse);
    }
  };

  const firstResponse = result?.responses?.[0];
  const skillGap = firstResponse?.skill_gap;
  const jobs = Array.isArray(firstResponse?.jobs)
    ? firstResponse.jobs
    : Array.isArray((firstResponse as any)?.jobs?.jobs)
      ? (firstResponse as any).jobs.jobs
      : [];
  const salary = firstResponse?.salary ?? firstResponse?.salary_benchmark;
  const learningPath = firstResponse?.learning_path;
  const learningPhases = Array.isArray(learningPath?.phases)
    ? learningPath.phases
    : [];
  const resumeAnalysis = firstResponse?.resume_analysis;
  const summary = firstResponse?.summary || result?.message || "";

  return (
    <div className="space-y-8">
      <PageHeader
        title="Career Dashboard"
        description="Skills, roadmaps, job intelligence, and AI career guidance"
        badge="Career"
      />

      <div className="grid gap-4 sm:grid-cols-2">
        <MetricCard
          title="Skills Tracked"
          value={skills.length}
          subtitle="Saved current skills"
          icon={BookOpen}
          gradient="from-emerald-500 to-teal-500"
        />
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-4">
          <div>
            <CardTitle className="text-base">Career Agent</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Run the backend career-agent workflow for skill gaps, jobs,
              salary, learning path, and resume analysis.
            </p>
          </div>
          <Button
            variant="gradient"
            onClick={handleRunCareerAgent}
            disabled={queryMutation.isPending}
          >
            {queryMutation.isPending ? "Running..." : "Run Career Agent"}
          </Button>
        </CardHeader>
        <CardContent>
          {queryMutation.isError && (
            <p className="text-sm text-destructive">
              {getErrorMessage(queryMutation.error)}
            </p>
          )}

          {firstResponse && (
            <div className="space-y-6">
              {summary && (
                <Card className="border-primary/20">
                  <CardHeader>
                    <CardTitle className="text-base">Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="whitespace-pre-line text-sm text-muted-foreground">
                      {summary}
                    </p>
                  </CardContent>
                </Card>
              )}

              {skillGap && (
                <Card className="border-primary/20">
                  <CardHeader>
                    <CardTitle className="text-base">Skill Gap</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">
                      {JSON.stringify(skillGap, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              )}

              <div className="grid gap-6 lg:grid-cols-2">
                <Card className="border-primary/20">
                  <CardHeader>
                    <CardTitle className="text-base">Job Matches</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {jobs.length > 0 ? (
                      jobs.map((job, idx) => (
                        <div
                          key={`${job.title}-${idx}`}
                          className="rounded-lg border p-3"
                        >
                          <p className="font-semibold">{job.title}</p>
                          <p className="text-sm text-muted-foreground">
                            {job.company}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {job.location}
                          </p>
                          {typeof job.embedding_match_score === "number" && (
                            <p className="mt-2 text-xs text-primary">
                              Match score: {job.embedding_match_score}%
                            </p>
                          )}
                          {job.description && (
                            <p className="mt-2 text-xs text-muted-foreground">
                              {job.description}
                            </p>
                          )}
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        No jobs returned.
                      </p>
                    )}
                  </CardContent>
                </Card>

                <Card className="border-primary/20">
                  <CardHeader>
                    <CardTitle className="text-base">
                      Salary Benchmark
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">
                      {JSON.stringify(salary, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              </div>

              <Card className="border-primary/20">
                <CardHeader>
                  <CardTitle className="text-base">Learning Path</CardTitle>
                </CardHeader>
                <CardContent>
                  {learningPath?.error ? (
                    <p className="text-sm text-destructive">
                      {learningPath.error}
                    </p>
                  ) : learningPhases.length > 0 ? (
                    <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">
                      {JSON.stringify(learningPath, null, 2)}
                    </pre>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No learning path returned.
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card className="border-primary/20">
                <CardHeader>
                  <CardTitle className="text-base">Resume Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  {resumeAnalysis ? (
                    <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">
                      {JSON.stringify(resumeAnalysis, null, 2)}
                    </pre>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No resume analysis returned.
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Skill Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Skill progress history will appear when real assessment history
                is available.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Career Roadmap</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Career roadmap will appear when personalized career planning is
                available.
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Skills</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {skills.length > 0 ? (
                skills.map((skill) => (
                  <p key={skill} className="text-sm font-medium">
                    {skill}
                  </p>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">
                  Add current skills to see them here.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {currentSalary !== null ? (
          <StatCard
            label="Current Salary"
            value={formatCurrency(currentSalary)}
            icon={DollarSign}
            iconColor="text-blue-500"
          />
        ) : null}
        <StatCard
          label="Target Role"
          value={targetRoleLabel}
          change={targetRoleChange}
          icon={TrendingUp}
          iconColor="text-emerald-500"
        />
        <StatCard
          label="Resume Tips"
          value={resumeTip}
          change={
            profile?.career?.resume
              ? "Resume profile detected"
              : "Complete your profile"
          }
          icon={BookOpen}
          iconColor="text-purple-500"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Job Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Job recommendations will appear when a real recommendations source
            is available.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
