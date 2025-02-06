"use client"

import { useState } from "react"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from "chart.js"
import { Line, Bar, Doughnut } from "react-chartjs-2"
import { ArrowUpIcon, ArrowDownIcon } from "@heroicons/react/24/solid"

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend)

// Add type definitions for our insights data
type InsightsSummary = {
  total_transactions: number
  date_range: {
    start: string
    end: string
  }
  total_credits: number
  total_debits: number
  net_change: number
}

type FrequentTransaction = {
  description: string
  frequency: number
  average_amount: number
  total_amount: number
}

type SpendingInsights = {
  average_transaction: number
  largest_credit: number
  largest_debit: number
  average_deposit: number
}

type BalanceTrend = {
  Date: string
  Balance: number
}

type CategorySpending = {
  [key: string]: number
}

type DailyCreditDebit = {
  Date: string
  Credit: number
  Debit: number
}

type BalanceTrends = {
  balance_over_time: BalanceTrend[]
  category_spending: CategorySpending
  daily_credits_vs_debits: DailyCreditDebit[]
  category_transaction_counts: CategorySpending
}

interface LoanFitness {
  score: number;
  max_score: number;
  risk_level: string;
  component_scores: {
    balance_stability: number;
    income_consistency: number;
    spending_discipline: number;
    savings_ratio: number;
  };
  interpretation: {
    balance_stability: string;
    income_consistency: string;
    spending_discipline: string;
    savings_ratio: string;
  };
}

type Insights = {
  summary: InsightsSummary
  frequent_transactions: FrequentTransaction[]
  spending_insights: SpendingInsights
  balance_trends: BalanceTrends
  loan_fitness: LoanFitness
}

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null)
  const [insights, setInsights] = useState<Insights | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(amount)
  }

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a PDF file first")
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append("pdf", file)

    try {
      const response = await fetch("http://localhost:8000/api/upload/", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Upload failed")
      }

      const data = await response.json()
      setInsights(data.insights)
    } catch (err) {
      setError("Failed to upload and process PDF. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
      },
      y: {
        grid: {
          color: "#f0f0f0",
        },
        ticks: {
          callback: function(tickValue: string | number) {
            return typeof tickValue === 'number' ? formatCurrency(tickValue) : tickValue;
          }
        },
      },
    },
  } as const;

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Casca AI - Rahil's MVP (Future S25 Intern) :)</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        {/* File Upload Section */}
        <div className="mb-8 bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-center w-full">
            <label
              htmlFor="dropzone-file"
              className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
            >
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg
                  className="w-10 h-10 mb-3 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  ></path>
                </svg>
                <p className="mb-2 text-sm text-gray-500">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-500">PDF file (MAX. 10MB)</p>
              </div>
              <input id="dropzone-file" type="file" className="hidden" onChange={handleFileChange} accept=".pdf" />
            </label>
          </div>
          {file && <p className="mt-2 text-sm text-gray-500">Selected file: {file.name}</p>}
          {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className={`mt-4 w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              !file || loading
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            }`}
          >
            {loading ? "Processing..." : "Analyze Statement"}
          </button>
        </div>

        {insights && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
              <SummaryCard
                title="Total Transactions"
                value={insights.summary.total_transactions}
                subtitle={`${insights.summary.date_range.start} to ${insights.summary.date_range.end}`}
              />
              <SummaryCard title="Total Credits" value={formatCurrency(insights.summary.total_credits)} color="green" />
              <SummaryCard title="Total Debits" value={formatCurrency(insights.summary.total_debits)} color="red" />
              <SummaryCard
                title="Net Change"
                value={formatCurrency(insights.summary.net_change)}
                color={insights.summary.net_change >= 0 ? "green" : "red"}
              />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <ChartCard title="Balance Over Time">
                <Line
                  options={chartOptions}
                  data={{
                    labels: insights.balance_trends.balance_over_time.map((point) =>
                      new Date(point.Date).toLocaleDateString(),
                    ),
                    datasets: [
                      {
                        label: "Balance",
                        data: insights.balance_trends.balance_over_time.map((point) => point.Balance),
                        borderColor: "rgb(59, 130, 246)",
                        backgroundColor: "rgba(59, 130, 246, 0.1)",
                        tension: 0.4,
                        fill: true,
                      },
                    ],
                  }}
                />
              </ChartCard>

              <ChartCard title="Daily Credits vs Debits">
                <Bar
                  options={{
                    ...chartOptions,
                    scales: {
                      x: {
                        stacked: true,
                        grid: {
                          display: false,
                        },
                      },
                      y: {
                        stacked: true,
                        grid: {
                          color: "rgba(0, 0, 0, 0.1)",
                        },
                        ticks: {
                          callback: function(value: string | number) {
                            if (typeof value === "number") {
                              return `$${value.toLocaleString()}`;
                            }
                            return value;
                          }
                        }
                      },
                    },
                  }}
                  data={{
                    labels: insights.balance_trends.daily_credits_vs_debits.map((day) =>
                      new Date(day.Date).toLocaleDateString(),
                    ),
                    datasets: [
                      {
                        label: "Credits",
                        data: insights.balance_trends.daily_credits_vs_debits.map((day) => day.Credit),
                        backgroundColor: "rgba(34, 197, 94, 0.6)",
                      },
                      {
                        label: "Debits",
                        data: insights.balance_trends.daily_credits_vs_debits.map((day) => -day.Debit),
                        backgroundColor: "rgba(239, 68, 68, 0.6)",
                      },
                    ],
                  }}
                />
              </ChartCard>
            </div>

            {/* Category Spending */}
            <ChartCard title="Spending by Category">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="h-64">
                  <Doughnut
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: "right",
                        },
                      },
                    }}
                    data={{
                      labels: Object.keys(insights.balance_trends.category_spending),
                      datasets: [
                        {
                          data: Object.values(insights.balance_trends.category_spending),
                          backgroundColor: [
                            "rgba(255, 99, 132, 0.6)",
                            "rgba(54, 162, 235, 0.6)",
                            "rgba(255, 206, 86, 0.6)",
                            "rgba(75, 192, 192, 0.6)",
                            "rgba(153, 102, 255, 0.6)",
                          ],
                        },
                      ],
                    }}
                  />
                </div>
                <div className="space-y-2">
                  {Object.entries(insights.balance_trends.category_spending).map(([category, amount]) => (
                    <div key={category} className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-600">{category}</span>
                      <span className="text-sm font-semibold text-gray-900">{formatCurrency(amount)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </ChartCard>

            {/* Frequent Transactions */}
            <ChartCard title="Frequent Transactions">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Description
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Frequency
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Average Amount
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Total Amount
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {insights.frequent_transactions.map((transaction, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {transaction.description}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{transaction.frequency}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatCurrency(transaction.average_amount)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatCurrency(transaction.total_amount)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ChartCard>

            {/* Loan Fitness Card */}
            <div className="mt-5">
              <LoanFitnessCard loanFitness={insights.loan_fitness} />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

function SummaryCard({
  title,
  value,
  subtitle,
  color = "blue",
}: { title: string; value: string | number; subtitle?: string; color?: "blue" | "green" | "red" }) {
  const colorClasses = {
    blue: "text-blue-600",
    green: "text-green-600",
    red: "text-red-600",
  }

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="p-5">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            {color === "green" && <ArrowUpIcon className="h-6 w-6 text-green-400" />}
            {color === "red" && <ArrowDownIcon className="h-6 w-6 text-red-400" />}
          </div>
          <div className="ml-5 w-0 flex-1">
            <dl>
              <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
              <dd className={`text-lg font-semibold ${colorClasses[color]}`}>{value}</dd>
              {subtitle && <dd className="text-sm text-gray-500">{subtitle}</dd>}
            </dl>
          </div>
        </div>
      </div>
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="p-5">
        <h3 className="text-lg leading-6 font-medium text-gray-900">{title}</h3>
        <div className="mt-5 h-80">{children}</div>
      </div>
    </div>
  )
}

function LoanFitnessCard({ loanFitness }: { loanFitness: LoanFitness }) {
  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low':
        return 'text-green-600';
      case 'medium':
        return 'text-yellow-600';
      case 'high':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 75) return 'text-green-600';
    if (score >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4 text-black">Loan Fitness Assessment</h2>
      <div className="flex items-center justify-between mb-6">
        <div>
          <span className={`text-4xl font-bold ${getScoreColor(loanFitness.score)}`}>
            {loanFitness.score}
          </span>
          <span className="text-gray-500 text-sm ml-1">/ {loanFitness.max_score}</span>
        </div>
        <div className={`text-lg font-semibold ${getRiskColor(loanFitness.risk_level)}`}>
          {loanFitness.risk_level} Risk
        </div>
      </div>

      <div className="space-y-4">
        {Object.entries(loanFitness.component_scores).map(([key, score]) => (
          <div key={key}>
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-gray-700">
                {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
              </span>
              <span className="text-sm text-gray-600">{score.toFixed(1)} / 25</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 rounded-full h-2"
                style={{ width: `${(score / 25) * 100}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">{loanFitness.interpretation[key]}</p>
          </div>
        ))}
      </div>
    </div>
  );
}