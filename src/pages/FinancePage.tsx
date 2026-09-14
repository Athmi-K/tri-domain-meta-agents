import {
  ArrowDownRight,
  ArrowUpRight,
  PiggyBank,
  TrendingUp,
  Wallet,
} from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { MetricCard } from "@/components/common/MetricCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useNavigate } from "react-router-dom";
import { useProfile } from "@/hooks";
import { Button } from "@/components/ui/button";
import { ROUTES } from "@/utils/constants";
import { formatCurrency, formatPercent } from "@/utils";
import { buildFinancePageData } from "@/utils/profileInsights";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQueryMutation } from "@/hooks";
import type {
  FinanceToolOutput,
  QueryRequest,
  QueryResponse,
} from "@/types";

type FinanceToolName =
  | "budget"
  | "savings"
  | "investments"
  | "debt"
  | "retirement"
  | "tax"
  | "financial_analysis";

const formatFinanceCurrency = (value: number | null | undefined) =>
  value == null ? "Not provided" : formatCurrency(value);

const formatFinancePercent = (value: number | null | undefined) =>
  value == null ? "Not provided" : formatPercent(value, 1);

const getFinanceOutput = (
  result: QueryResponse | undefined,
  tool: FinanceToolName,
): FinanceToolOutput | undefined => {
  const response = result?.responses?.find((item) => item.domain === "finance");
  if (tool === "investments") {
    return response?.tool_outputs?.investments ?? response?.tool_outputs?.investment;
  }
  return response?.tool_outputs?.[tool];
};

export function FinancePage() {
  const { data: profile, isLoading: isProfileLoading } = useProfile();
  const navigate = useNavigate();
  const financeData = useMemo(() => buildFinancePageData(profile), [profile]);
  const {
    monthlyIncome,
    monthlyExpenses,
    savings,
    savingsRate,
    riskProfile,
  } = financeData;
  const queryMutation = useQueryMutation();
  const [analysisResults, setAnalysisResults] = useState<
    Partial<Record<FinanceToolName, QueryResponse>>
  >({});
  const [analysisErrors, setAnalysisErrors] = useState<
    Partial<Record<FinanceToolName, string>>
  >({});

  const runFinanceQuery = useCallback(
    async (tool: FinanceToolName, query: string) => {
      try {
        const res = await queryMutation.mutateAsync({
          name: "User",
          age: profile?.general?.age ?? 0,
          query,
          domain: "finance",
          monthly_income: monthlyIncome ?? 0,
          monthly_expenses: monthlyExpenses ?? 0,
          current_savings: profile?.finance?.current_savings,
          savings_goal: profile?.finance?.savings_goal,
          portfolio: profile?.finance?.portfolio,
          risk_tolerance: riskProfile ?? undefined,
          investment_experience:
            profile?.finance?.investment_experience,
          financial_goals: profile?.finance?.financial_goals,
          debts: profile?.finance?.debts,
          total_debt: profile?.finance?.total_debt,
          monthly_debt_payment: profile?.finance?.monthly_debt_payment,
          retirement_age: profile?.finance?.retirement_age,
          retirement_savings: profile?.finance?.retirement_savings,
          monthly_contribution: profile?.finance?.monthly_contribution,
          annual_income: profile?.finance?.annual_income,
          tax_deductions: profile?.finance?.tax_deductions,
        } as QueryRequest);

        setAnalysisResults((previous) => ({
          ...previous,
          [tool]: res,
        }));
        setAnalysisErrors((previous) => {
          const next = { ...previous };
          delete next[tool];
          return next;
        });
      } catch (err) {
        console.error(`Finance ${tool} analysis failed:`, err);
        setAnalysisErrors((previous) => ({
          ...previous,
          [tool]: "This analysis could not be generated right now.",
        }));
      }
    },
    [
      queryMutation,
      profile?.general?.age,
      profile?.finance?.savings_goal,
      profile?.finance?.current_savings,
      profile?.finance?.portfolio,
      profile?.finance?.investment_experience,
      profile?.finance?.financial_goals,
      profile?.finance?.debts,
      profile?.finance?.total_debt,
      profile?.finance?.monthly_debt_payment,
      profile?.finance?.retirement_age,
      profile?.finance?.retirement_savings,
      profile?.finance?.monthly_contribution,
      profile?.finance?.annual_income,
      profile?.finance?.tax_deductions,
      monthlyIncome,
      monthlyExpenses,
      riskProfile,
    ],
  );

  const autoAnalysisKey = [
    profile?.general?.age ?? "",
    monthlyIncome ?? "",
    monthlyExpenses ?? "",
    profile?.finance?.savings_goal ?? "",
    profile?.finance?.current_savings ?? "",
    JSON.stringify(profile?.finance?.portfolio ?? {}),
    riskProfile ?? "",
    profile?.finance?.investment_experience ?? "",
    profile?.finance?.financial_goals ?? "",
    JSON.stringify(profile?.finance?.debts ?? []),
    profile?.finance?.total_debt ?? "",
    profile?.finance?.monthly_debt_payment ?? "",
    profile?.finance?.retirement_age ?? "",
    profile?.finance?.retirement_savings ?? "",
    profile?.finance?.monthly_contribution ?? "",
    profile?.finance?.annual_income ?? "",
    JSON.stringify(profile?.finance?.tax_deductions ?? {}),
  ].join("|");

  const hasDebtInputs = Boolean(
    (profile?.finance?.total_debt != null &&
      profile.finance.total_debt > 0) ||
      (profile?.finance?.debts && profile.finance.debts.length > 0) ||
      (profile?.finance?.monthly_debt_payment != null &&
        profile.finance.monthly_debt_payment > 0),
  );
  const hasRetirementInputs = Boolean(
    profile?.general?.age != null &&
      monthlyIncome != null &&
      monthlyExpenses != null &&
      profile?.finance?.retirement_age != null &&
      profile?.finance?.retirement_savings != null &&
      profile?.finance?.monthly_contribution != null,
  );

  const lastAnalyzedKey = useRef<string | null>(null);

  useEffect(() => {
    if (isProfileLoading || !profile) return;

    if (lastAnalyzedKey.current === autoAnalysisKey) return;

    const runInitialAnalyses = async () => {
      const requests: Promise<void>[] = [];

      if (monthlyIncome != null && monthlyExpenses != null) {
        requests.push(
          runFinanceQuery(
            "financial_analysis",
            "Analyze my financial strengths and weaknesses and provide personalized financial recommendations.",
          ),
        );

        requests.push(
          runFinanceQuery(
            "budget",
            "Analyze my monthly budget and spending and provide personalized recommendations.",
          ),
        );

        if (profile.finance?.savings_goal != null) {
          requests.push(
            runFinanceQuery(
              "savings",
              "Analyze my savings progress, savings goal, and provide a personalized plan to reach the goal.",
            ),
          );
        }

        if (
          profile.general?.age != null &&
          riskProfile &&
          profile.finance?.investment_experience &&
          profile.finance?.financial_goals
        ) {
          requests.push(
            runFinanceQuery(
              "investments",
              "Analyze my investment profile and provide personalized investment allocation guidance.",
            ),
          );
        }

      }

      if (hasDebtInputs) {
        requests.push(
          runFinanceQuery(
            "debt",
            "Analyze my debt and recommend a repayment strategy.",
          ),
        );
      }

      if (hasRetirementInputs) {
        requests.push(
          runFinanceQuery(
            "retirement",
            "Analyze my retirement plan and projected corpus.",
          ),
        );
      }

      if (profile.finance?.annual_income != null || monthlyIncome != null) {
        requests.push(
          runFinanceQuery(
            "tax",
            "Analyze my income tax situation and compare the available tax regimes.",
          ),
        );
      }

      if (requests.length > 0) {
        lastAnalyzedKey.current = autoAnalysisKey;
        await Promise.allSettled(requests);
      }
    };

    void runInitialAnalyses();
  }, [
    isProfileLoading,
    profile,
    autoAnalysisKey,
    runFinanceQuery,
    hasDebtInputs,
    hasRetirementInputs,
  ]);

  const budgetData = getFinanceOutput(analysisResults.budget, "budget");
  const savingsData = getFinanceOutput(analysisResults.savings, "savings");
  const investmentData = getFinanceOutput(
    analysisResults.investments,
    "investments",
  );
  const taxData = getFinanceOutput(analysisResults.tax, "tax");
  const financialAnalysisData = getFinanceOutput(
    analysisResults.financial_analysis,
    "financial_analysis",
  );
  const debtData = getFinanceOutput(analysisResults.debt, "debt");
  const retirementData = getFinanceOutput(
    analysisResults.retirement,
    "retirement",
  );
  const overviewSavings = financialAnalysisData?.monthly_savings ?? savings;
  const overviewSavingsRate =
    financialAnalysisData?.savings_rate_pct ?? savingsRate;
  const profileCurrentSavings = profile?.finance?.current_savings;
  const profileSavingsGoal = profile?.finance?.savings_goal;
  const currentSavings =
    savingsData?.current_savings ?? profileCurrentSavings;
  const remainingSavings =
    savingsData?.remaining_to_goal ??
    savingsData?.shortfall ??
    (profileSavingsGoal != null && profileCurrentSavings != null
      ? Math.max(0, profileSavingsGoal - profileCurrentSavings)
      : null);
  const profileTotalDebt = profile?.finance?.total_debt;
  const profileMonthlyDebtPayment = profile?.finance?.monthly_debt_payment;
  const debtTotal = debtData?.total_debt ?? profileTotalDebt;
  const debtMonthlyPayment =
    debtData?.monthly_payment ?? profileMonthlyDebtPayment;
  const debtToIncome =
    debtData?.debt_to_income_ratio ??
    (debtMonthlyPayment != null && monthlyIncome != null && monthlyIncome > 0
      ? (debtMonthlyPayment / monthlyIncome) * 100
      : null);
  const hasInvestmentProfile = Boolean(
    profile?.finance?.investments ||
      riskProfile ||
      profile?.finance?.investment_experience ||
      profile?.finance?.financial_goals,
  );
  const hasPortfolioData = Boolean(
    profile?.finance?.portfolio &&
      Object.keys(profile.finance.portfolio).length > 0,
  );
  const hasSavingsProfile = profile?.finance?.savings_goal != null;
  const hasTaxProfile =
    profile?.finance?.annual_income != null || monthlyIncome != null;
  const canRunSavings =
    monthlyIncome != null && monthlyExpenses != null && hasSavingsProfile;
  const canRunInvestment = Boolean(
    profile?.general?.age != null &&
      monthlyIncome != null &&
    hasPortfolioData &&
      riskProfile &&
      profile?.finance?.investment_experience &&
      profile?.finance?.financial_goals,
  );
  if (isProfileLoading) {
    return (
      <div className="space-y-8">
        <PageHeader
          title="Finance Dashboard"
          description="Loading profile..."
        />
        <div className="flex justify-center py-12">
          <div className="loader" />
        </div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="space-y-8">
        <PageHeader
          title="Finance Dashboard"
          description="Complete your profile to see financial insights"
          badge="Profile Needed"
        />
        <Card>
          <CardContent className="p-8 text-center">
            <h3 className="text-lg font-semibold mb-2">
              No finance profile yet
            </h3>
            <p className="text-sm text-muted-foreground mb-4">
              Add your income and expenses in the profile to view personalized
              budgets and recommendations.
            </p>
            <Button variant="gradient" onClick={() => navigate(ROUTES.PROFILE)}>
              Edit Profile
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader
        title="Finance Dashboard"
        description="Budget tracking, savings, and investment guidance"
        badge="Finance"
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <MetricCard
          title="Monthly Income"
          value={
            monthlyIncome != null
              ? formatCurrency(monthlyIncome)
              : "Not provided"
          }
          subtitle="Gross earnings"
          icon={ArrowUpRight}
          gradient="from-emerald-500 to-teal-500"
        />
        <MetricCard
          title="Monthly Expenses"
          value={
            monthlyExpenses != null
              ? formatCurrency(monthlyExpenses)
              : "Not provided"
          }
          subtitle="Total spending"
          icon={ArrowDownRight}
          gradient="from-rose-500 to-pink-500"
        />
        <MetricCard
          title="Savings"
          value={
            overviewSavings != null
              ? formatCurrency(overviewSavings)
              : "Not provided"
          }
          subtitle={
            overviewSavingsRate != null
              ? `${formatPercent(overviewSavingsRate, 1)} savings rate`
              : "Add income & expenses"
          }
          icon={PiggyBank}
          gradient="from-blue-500 to-indigo-500"
        />
        <MetricCard
          title="Savings Rate"
          value={
            overviewSavingsRate != null
              ? formatPercent(overviewSavingsRate, 1)
              : "Not provided"
          }
          subtitle="Of gross income"
          icon={TrendingUp}
          gradient="from-violet-500 to-purple-500"
        />
        <MetricCard
          title="Financial Health"
          value={
            typeof financialAnalysisData?.health_score === "number"
              ? `${financialAnalysisData.health_score}/100`
              : "Generating..."
          }
          subtitle={financialAnalysisData?.health_status ?? "Finance Agent score"}
          icon={TrendingUp}
          gradient="from-cyan-500 to-blue-500"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Income vs Expenses</CardTitle>
        </CardHeader>
        <CardContent>
          {monthlyIncome != null && monthlyExpenses != null ? (
            <div className="flex h-52 items-end justify-center gap-8 border-b border-border pt-6">
              {[
                {
                  label: "Income",
                  value: monthlyIncome,
                  color: "bg-emerald-500",
                },
                {
                  label: "Expenses",
                  value: monthlyExpenses,
                  color: "bg-rose-500",
                },
                {
                  label: "Savings",
                  value: overviewSavings ?? 0,
                  color: "bg-blue-500",
                },
              ].map((row) => {
                const max = monthlyIncome || 1;
                const heightPct = Math.max(
                  4,
                  Math.min(100, (Math.abs(row.value) / max) * 100),
                );
                return (
                  <div
                    key={row.label}
                    className="flex h-full w-20 flex-col items-center justify-end gap-2"
                  >
                    <span className="text-sm text-muted-foreground">
                      {formatCurrency(row.value)}
                    </span>
                    <div
                      className={`w-full rounded-t-md ${row.color}`}
                      style={{ height: `${heightPct}%` }}
                    >
                      <span className="sr-only">{row.label}</span>
                    </div>
                    <span className="text-sm font-medium">{row.label}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              Add your monthly income and expenses to your profile to see this
              comparison.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Financial Health Analysis
          </CardTitle>
        </CardHeader>

        <CardContent>
          {financialAnalysisData ? (
            <div className="space-y-6">
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="rounded-xl border p-4">
                  <p className="text-sm text-muted-foreground">
                    Financial Health Score
                  </p>
                  <p className="mt-1 text-2xl font-semibold">
                    {typeof financialAnalysisData.health_score === "number"
                      ? `${financialAnalysisData.health_score}/100`
                      : "Not available"}
                  </p>
                </div>

                <div className="rounded-xl border p-4">
                  <p className="text-sm text-muted-foreground">
                    Financial Status
                  </p>
                  <p className="mt-1 text-xl font-semibold capitalize">
                    {typeof financialAnalysisData.health_status === "string"
                      ? financialAnalysisData.health_status
                      : "Not available"}
                  </p>
                </div>

                <div className="rounded-xl border p-4">
                  <p className="text-sm text-muted-foreground">
                    Expense Ratio
                  </p>
                  <p className="mt-1 text-xl font-semibold">
                    {typeof financialAnalysisData.expense_ratio_pct === "number"
                      ? formatPercent(
                          financialAnalysisData.expense_ratio_pct,
                          1,
                        )
                      : "Not available"}
                  </p>
                </div>
              </div>

              {Array.isArray(financialAnalysisData.strengths) &&
                financialAnalysisData.strengths.length > 0 && (
                  <div>
                    <h4 className="mb-2 text-sm font-semibold">
                      Financial Strengths
                    </h4>

                    <ul className="space-y-2">
                      {financialAnalysisData.strengths.map(
                        (item: unknown, index: number) => (
                          <li
                            key={index}
                            className="text-sm text-muted-foreground"
                          >
                            • {String(item)}
                          </li>
                        ),
                      )}
                    </ul>
                  </div>
                )}

              {Array.isArray(financialAnalysisData.weaknesses) &&
                financialAnalysisData.weaknesses.length > 0 && (
                  <div>
                    <h4 className="mb-2 text-sm font-semibold">
                      Areas to Improve
                    </h4>

                    <ul className="space-y-2">
                      {financialAnalysisData.weaknesses.map(
                        (item: unknown, index: number) => (
                          <li
                            key={index}
                            className="text-sm text-muted-foreground"
                          >
                            • {String(item)}
                          </li>
                        ),
                      )}
                    </ul>
                  </div>
                )}

              {financialAnalysisData.recommendation && (
                <div className="rounded-xl bg-muted/50 p-4">
                  <p className="mb-1 text-sm font-semibold">
                    Personalized Recommendation
                  </p>

                  <p className="text-sm text-muted-foreground">
                    {financialAnalysisData.recommendation}
                  </p>
                </div>
              )}
            </div>
          ) : analysisErrors.financial_analysis ? (
            <p className="py-8 text-center text-sm text-destructive">
              {analysisErrors.financial_analysis}
            </p>
          ) : (
            <div className="py-8 text-center">
              <p className="text-sm font-medium">
                Generating your financial analysis...
              </p>

              <p className="mt-1 text-sm text-muted-foreground">
                The Finance Agent is analyzing your income, expenses, savings,
                and financial profile.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Budget Breakdown</CardTitle>
          </CardHeader>

          <CardContent>
            {budgetData ? (
              <div className="space-y-6">
                <div className="flex items-center justify-between rounded-xl border p-4">
                  <div>
                    <p className="text-sm font-medium">Spending Status</p>
                    <p className="text-sm text-muted-foreground">
                      {typeof financialAnalysisData?.expense_ratio_pct === "number"
                        ? `Expense ratio: ${formatPercent(financialAnalysisData.expense_ratio_pct, 1)}`
                        : "Based on your recorded budget data"}
                    </p>
                  </div>
                  {budgetData.savings_status && (
                    <Badge variant="secondary">{budgetData.savings_status}</Badge>
                  )}
                </div>

                {/* Category Breakdown */}
                {budgetData.category_breakdown &&
                  budgetData.category_breakdown.length > 1 && (
                  <div>
                    <h4 className="mb-3 text-sm font-semibold">
                      Expense Allocation
                    </h4>

                    <div className="space-y-3">
                      {budgetData.category_breakdown.map(
                        (category: {
                          category: string;
                          amount: number;
                          share_pct: number;
                          status: string;
                        }) => (
                          <div key={category.category}>
                            <div className="mb-1 flex items-center justify-between text-sm">
                              <span className="font-medium capitalize">
                                {category.category}
                              </span>

                              <span className="text-muted-foreground">
                                {formatCurrency(category.amount)} (
                                  {formatPercent(category.share_pct, 1)})
                              </span>
                            </div>

                            <div className="h-2.5 w-full rounded-full bg-muted">
                              <div
                                className="h-2.5 rounded-full bg-primary"
                                style={{
                                  width: `${Math.min(
                                    100,
                                    Math.max(0, category.share_pct),
                                  )}%`,
                                }}
                              />
                            </div>
                          </div>
                        ),
                      )}
                    </div>
                  </div>
                )}

                {budgetData.category_breakdown?.length === 1 && (
                  <p className="text-sm text-muted-foreground">
                    Recorded spending: {formatCurrency(budgetData.category_breakdown[0].amount)}.
                  </p>
                )}

                {/* Overspending */}
                {budgetData.overspending && budgetData.overspending.length > 0 && (
                  <div className="rounded-xl border border-destructive/30 p-4">
                    <p className="mb-2 text-sm font-semibold">
                      Overspending Alerts
                    </p>

                    <div className="space-y-2">
                      {budgetData.overspending.map(
                        (item: unknown, index: number) => (
                          <p
                            key={index}
                            className="text-sm text-muted-foreground"
                          >
                            {typeof item === "string"
                              ? item
                              : JSON.stringify(item)}
                          </p>
                        ),
                      )}
                    </div>
                  </div>
                )}

                {/* Backend Recommendation */}
                {budgetData.recommendation && (
                  <div className="rounded-xl bg-muted/50 p-4">
                    <p className="mb-1 text-sm font-semibold">Recommendation</p>
                    <p className="text-sm text-muted-foreground">
                      {budgetData.recommendation}
                    </p>
                  </div>
                )}
              </div>
            ) : analysisErrors.budget ? (
              <p className="py-8 text-center text-sm text-destructive">
                {analysisErrors.budget}
              </p>
            ) : monthlyIncome != null && monthlyExpenses != null ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <PiggyBank className="mb-3 h-8 w-8 text-muted-foreground" />

                <p className="text-sm font-medium">
                  Generating personalized budget insights...
                </p>

                <p className="mt-1 text-sm text-muted-foreground">
                  The Finance Agent is analyzing your income and expenses.
                </p>
              </div>
            ) : (
              <p className="py-8 text-center text-sm text-muted-foreground">
                Add monthly income and expenses to view personalized budget insights.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Wallet className="h-4 w-4 text-primary" />
              Portfolio Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            {hasPortfolioData && canRunInvestment ? (
              <div className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-4">
                  <div className="rounded-xl border p-4">
                    <p className="text-sm text-muted-foreground">Investments</p>
                    <p className="mt-1 text-sm font-medium">
                      {profile.finance?.investments || "Profile details available"}
                    </p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-sm text-muted-foreground">Risk Tolerance</p>
                    <p className="mt-1 text-sm font-medium">
                      {riskProfile || "Not provided"}
                    </p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-sm text-muted-foreground">Experience</p>
                    <p className="mt-1 text-sm font-medium">
                      {profile.finance?.investment_experience || "Not provided"}
                    </p>
                  </div>
                </div>
                {investmentData ? (
                  <div className="space-y-3 text-sm">
                    {investmentData.portfolio_value != null && (
                      <p>
                        Portfolio value: {formatCurrency(investmentData.portfolio_value)}
                      </p>
                    )}
                    {investmentData.target_allocation && (
                      <p className="text-muted-foreground">
                        Target allocation: {Object.entries(investmentData.target_allocation)
                          .map(([asset, percentage]) => `${asset} ${percentage}%`)
                          .join(", ")}
                      </p>
                    )}
                    {investmentData.recommendation && (
                      <p className="text-muted-foreground">{investmentData.recommendation}</p>
                    )}
                  </div>
                ) : analysisErrors.investments ? (
                  <p className="text-sm text-destructive">
                    {analysisErrors.investments}
                  </p>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Generating personalized investment insights...
                  </p>
                )}
              </div>
            ) : hasInvestmentProfile ? (
              <p className="text-sm text-muted-foreground">
                Add investment details to view personalized investment insights.
              </p>
            ) : (
              <p className="text-sm text-muted-foreground">
                Add investment, risk tolerance, or experience details to view portfolio insights.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Savings &amp; Goals</CardTitle>
          </CardHeader>
          <CardContent>
            {hasSavingsProfile && canRunSavings ? (
              <div className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-xl border p-4">
                    <p className="text-sm text-muted-foreground">Savings Goal</p>
                    <p className="mt-1 text-xl font-semibold">
                      {formatFinanceCurrency(profile.finance?.savings_goal)}
                    </p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-sm text-muted-foreground">Current Savings</p>
                    <p className="mt-1 text-xl font-semibold">
                      {formatFinanceCurrency(currentSavings)}
                    </p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-sm text-muted-foreground">Amount Remaining</p>
                    <p className="mt-1 text-xl font-semibold">
                      {formatFinanceCurrency(remainingSavings)}
                    </p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-sm text-muted-foreground">Savings Rate</p>
                    <p className="mt-1 text-xl font-semibold">
                      {savingsData
                        ? formatFinancePercent(savingsData.savings_rate_pct)
                        : "Generating..."}
                    </p>
                  </div>
                </div>
                {savingsData ? (
                  <>
                    {savingsData.months_to_goal != null && (
                      <p className="text-sm text-muted-foreground">
                        Estimated time to goal: {savingsData.months_to_goal} months.
                      </p>
                    )}
                    {savingsData.recommendation && (
                      <p className="text-sm text-muted-foreground">
                        {savingsData.recommendation}
                      </p>
                    )}
                  </>
                ) : analysisErrors.savings ? (
                  <p className="text-sm text-destructive">
                    {analysisErrors.savings}
                  </p>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Generating personalized savings insights...
                  </p>
                )}
              </div>
            ) : hasSavingsProfile ? (
              <p className="text-sm text-muted-foreground">
                Add monthly income and expenses to view personalized savings insights.
              </p>
            ) : (
              <p className="text-sm text-muted-foreground">
                Add a savings goal to view savings progress and time-to-goal insights.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Tax Insights</CardTitle>
          </CardHeader>
          <CardContent>
            {hasTaxProfile ? (
              <div className="space-y-4">
                {taxData ? (
                  <div className="space-y-4">
                    <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-xl border p-4">
                      <p className="text-sm text-muted-foreground">Annual Income</p>
                      <p className="mt-1 text-xl font-semibold">
                        {formatFinanceCurrency(taxData.gross_income)}
                      </p>
                    </div>
                    <div className="rounded-xl border p-4">
                      <p className="text-sm text-muted-foreground">Recommended Regime</p>
                      <p className="mt-1 text-xl font-semibold">
                        {taxData.recommended_regime || "Not provided"}
                      </p>
                    </div>
                    </div>
                    <div className="overflow-x-auto rounded-xl border">
                      <table className="w-full text-left text-sm">
                        <thead className="border-b bg-muted/50">
                          <tr>
                            <th className="px-4 py-3 font-medium">Regime</th>
                            <th className="px-4 py-3 font-medium">Taxable Income</th>
                            <th className="px-4 py-3 font-medium">Estimated Tax</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(["old", "new"] as const).map((regime) => {
                            const result = taxData[`${regime}_regime`];
                            return (
                              <tr key={regime} className="border-b last:border-0">
                                <td className="px-4 py-3 capitalize">{regime} regime</td>
                                <td className="px-4 py-3">
                                  {formatFinanceCurrency(
                                    typeof result?.taxable_income === "number"
                                      ? result.taxable_income
                                      : undefined,
                                  )}
                                </td>
                                <td className="px-4 py-3">
                                  {formatFinanceCurrency(
                                    typeof result?.tax_liability === "number"
                                      ? result.tax_liability
                                      : undefined,
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    {taxData.tax_savings_vs_other != null && (
                      <p className="text-sm text-muted-foreground">
                        Potential saving: {formatCurrency(taxData.tax_savings_vs_other)}
                      </p>
                    )}
                  </div>
                ) : analysisErrors.tax ? (
                  <p className="text-sm text-destructive">
                    {analysisErrors.tax}
                  </p>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Generating personalized tax insights...
                  </p>
                )}
                {taxData?.recommendation && (
                  <p className="text-sm text-muted-foreground">{taxData.recommendation}</p>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Add monthly income to view tax insights.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Debt Management</CardTitle>
          </CardHeader>
          <CardContent>
            {debtData || hasDebtInputs ? (
              <div className="space-y-3 text-sm">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-xl border p-4">
                    <p className="text-muted-foreground">Total Debt</p>
                    <p className="mt-1 text-xl font-semibold">{formatFinanceCurrency(debtTotal)}</p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-muted-foreground">Monthly Payment</p>
                    <p className="mt-1 text-xl font-semibold">{formatFinanceCurrency(debtMonthlyPayment)}</p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-muted-foreground">Debt-to-Income</p>
                    <p className="mt-1 text-xl font-semibold">
                      {formatFinancePercent(debtToIncome)}
                    </p>
                  </div>
                </div>
                {debtData?.recommended_strategy && (
                  <p className="text-sm font-medium capitalize">
                    Recommended strategy: {debtData.recommended_strategy}
                  </p>
                )}
                {debtData?.summary && <p className="text-muted-foreground">{debtData.summary}</p>}
              </div>
            ) : analysisErrors.debt ? (
              <p className="text-sm text-destructive">{analysisErrors.debt}</p>
            ) : hasDebtInputs ? (
              <p className="text-sm text-muted-foreground">Generating personalized debt insights...</p>
            ) : (
              <p className="text-sm text-muted-foreground">No debt information available.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Retirement Planning</CardTitle>
          </CardHeader>
          <CardContent>
            {retirementData?.error ? (
              <p className="text-sm text-destructive">{retirementData.error}</p>
            ) : retirementData ? (
              <div className="space-y-3 text-sm">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-xl border p-4">
                    <p className="text-muted-foreground">Projected Corpus</p>
                    <p className="mt-1 text-xl font-semibold">{formatFinanceCurrency(retirementData.projected_corpus)}</p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-muted-foreground">Required Corpus</p>
                    <p className="mt-1 text-xl font-semibold">{formatFinanceCurrency(retirementData.corpus_needed)}</p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="text-muted-foreground">Retirement Status</p>
                    <p className="mt-1 text-xl font-semibold capitalize">{retirementData.status || "Not available"}</p>
                  </div>
                </div>
                <p className="text-muted-foreground">
                  Monthly contribution: {formatFinanceCurrency(retirementData.monthly_contribution)}
                  {retirementData.required_monthly_contrib != null
                    ? ` · Required: ${formatCurrency(retirementData.required_monthly_contrib)}`
                    : ""}
                </p>
                {retirementData.recommendation && <p className="text-muted-foreground">{retirementData.recommendation}</p>}
              </div>
            ) : analysisErrors.retirement ? (
              <p className="text-sm text-destructive">{analysisErrors.retirement}</p>
            ) : hasRetirementInputs ? (
              <p className="text-sm text-muted-foreground">Generating personalized retirement insights...</p>
            ) : (
              <p className="text-sm text-muted-foreground">No retirement information available.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
