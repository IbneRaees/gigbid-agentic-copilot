import { expect, test } from "@playwright/test";

test.describe("GigBid Copilot Workbench E2E", () => {
  test("renders navigation tabs and job feed", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("GigBid")).toBeVisible();
    await expect(page.getByTestId("tab-jobs")).toBeVisible();
    await expect(page.getByTestId("tab-trace")).toBeVisible();
    await expect(page.getByTestId("tab-approvals")).toBeVisible();
    await expect(page.getByTestId("tab-conversations")).toBeVisible();
    await expect(page.getByTestId("tab-memory")).toBeVisible();
    await expect(page.getByTestId("tab-evals")).toBeVisible();
    await expect(page.getByTestId("tab-audit")).toBeVisible();
  });

  test("switches tabs cleanly and shows HITL approval empty state", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("tab-approvals").click();
    await expect(
      page.getByText(/Human-in-the-Loop Approval Queue|Awaiting Human Approval/i)
    ).toBeVisible();

    await page.getByTestId("tab-evals").click();
    await expect(page.getByText(/RAG Variant Benchmark/i)).toBeVisible();

    await page.getByTestId("tab-audit").click();
    await expect(page.getByText(/MCP Security & PII-Scrubbed Audit Log/i)).toBeVisible();
  });
});
