/**
 * Workflow 21: A/B Testing (Pool-based)
 *
 * Tests for the A/B testing pool editor, streaming comparison, vote buttons,
 * preference submission, admin gating, and metrics.
 *
 * NOTE: In the deployed build, #ab-settings-section lives inside
 * SECTION#settings-advanced which is hidden by default.  Tests that need to
 * *see* or *click* pool-editor elements call showPoolEditor() which force-
 * reveals every hidden ancestor so Playwright can interact with them.
 */
import {
  test,
  expect,
  setupBasicMocks,
  setupABAdminMocks,
  setupABAdminInactiveMocks,
  mockData,
  createABStreamResponse,
} from '../fixtures';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Open the settings modal. */
async function openSettings(page: import('@playwright/test').Page) {
  await page.getByRole('button', { name: /settings/i }).click();
  await expect(page.locator('.settings-modal')).toBeVisible();
}

/**
 * Open settings AND force every hidden ancestor of #ab-settings-section
 * visible so Playwright can interact with pool-editor elements.
 */
async function showPoolEditor(page: import('@playwright/test').Page) {
  await openSettings(page);
  // Wait until the pool response has been processed (badge text is set by loadABPool)
  await page.locator('#ab-pool-status').waitFor({ state: 'attached' });
  await page.evaluate(() => {
    const section = document.getElementById('ab-settings-section');
    if (!section) return;
    let el: HTMLElement | null = section;
    while (el) {
      if (el.hidden) el.hidden = false;
      if (getComputedStyle(el).display === 'none') {
        el.style.setProperty('display', 'block', 'important');
      }
      el = el.parentElement;
    }
  });
}

// =============================================================================
// Admin gating -- pool editor visibility
// =============================================================================

test.describe('A/B Pool Editor -- Admin Gating', () => {

  test('pool section display is none for non-admin users', async ({ page }) => {
    await setupBasicMocks(page);
    await page.goto('/chat');
    // Wait for pool data to settle (non-admin keeps display:none)
    await page.waitForTimeout(500);
    const display = await page.locator('#ab-settings-section').evaluate(
      (el: HTMLElement) => el.style.display,
    );
    expect(display).toBe('none');
  });

  test('pool section display is cleared for admin users', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');
    await expect(page.locator('#ab-pool-status')).toHaveText('Active');
    const display = await page.locator('#ab-settings-section').evaluate(
      (el: HTMLElement) => el.style.display,
    );
    expect(display).toBe('');
  });

  test('pool status badge shows Active when pool is enabled', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');
    await expect(page.locator('#ab-pool-status')).toHaveText('Active');
  });

  test('pool status badge shows Inactive when pool is disabled', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminInactiveMocks(page);
    await page.goto('/chat');
    await expect(page.locator('#ab-pool-status')).toHaveText('Inactive');
  });
});

// =============================================================================
// Pool editor -- agent list rendering
// =============================================================================

test.describe('A/B Pool Editor -- Agent List', () => {

  test('renders all agents including ab_only variants', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');

    const agentRows = page.locator('#ab-pool-agent-list .ab-pool-agent-row');
    await expect(agentRows).toHaveCount(mockData.agentsList.agents.length);
  });

  test('ab_only agents have an AB badge', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');

    const abBadges = page.locator('.ab-pool-ab-badge');
    await expect(abBadges).toHaveCount(2);
  });

  test('champion is pre-selected and marked', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');

    const championRow = page.locator('.ab-pool-agent-row.champion');
    await expect(championRow).toHaveCount(1);
    await expect(championRow).toHaveAttribute('data-agent', mockData.abPoolAdmin.champion!);
  });

  test('pool variants have checkboxes checked', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');

    const checkedBoxes = page.locator('.ab-pool-agent-row.selected input[type="checkbox"]:checked');
    await expect(checkedBoxes).toHaveCount(mockData.abPoolAdmin.variants!.length);
  });
});

// =============================================================================
// Pool editor -- save / disable interactions
// =============================================================================

test.describe('A/B Pool Editor -- Save and Disable', () => {

  test('save button is enabled when champion + 2+ variants selected', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');
    await expect(page.locator('#ab-pool-save')).toBeEnabled();
  });

  test('save button is disabled when fewer than 2 agents selected', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminInactiveMocks(page);
    await page.goto('/chat');
    await expect(page.locator('#ab-pool-save')).toBeDisabled();
  });

  test('disable button visible when pool is active', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');
    await showPoolEditor(page);
    await expect(page.locator('#ab-pool-disable')).toBeVisible();
  });

  test('disable button hidden when pool is inactive', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminInactiveMocks(page);
    await page.goto('/chat');
    await showPoolEditor(page);
    await expect(page.locator('#ab-pool-disable')).toBeHidden();
  });

  test('clicking save sends correct payload', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    let savedPayload: any = null;
    await page.route('**/api/ab/pool/set', async (route) => {
      const body = route.request().postDataJSON();
      savedPayload = body;
      await route.fulfill({ status: 200, json: { success: true, ...mockData.abPoolAdmin } });
    });

    await page.goto('/chat');
    await showPoolEditor(page);
    await page.locator('#ab-pool-save').click();

    // The "Pool saved" message flashes then gets cleared by _updateABPoolSaveState
    // so we verify the payload was sent correctly instead.
    await page.waitForTimeout(300);
    expect(savedPayload).toBeTruthy();
    expect(savedPayload.champion).toBe(mockData.abPoolAdmin.champion);
    expect(savedPayload.variants).toEqual(expect.arrayContaining(mockData.abPoolAdmin.variants!));
  });

  test('clicking disable calls endpoint and updates UI', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    let disableCalled = false;
    await page.route('**/api/ab/pool/disable', async (route) => {
      disableCalled = true;
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');
    await showPoolEditor(page);
    await page.locator('#ab-pool-disable').click();

    await expect(page.locator('#ab-pool-status')).toHaveText('Inactive');
    expect(disableCalled).toBe(true);
  });

  test('validation message when less than 2 agents selected', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminInactiveMocks(page);
    await page.goto('/chat');
    await showPoolEditor(page);

    const firstRow = page.locator('.ab-pool-agent-row').first();
    await firstRow.locator('input[type="checkbox"]').check({ force: true });

    await expect(page.locator('#ab-pool-message')).toContainText('Select at least 2 agents');
  });

  test('validation message when no champion designated', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminInactiveMocks(page);
    await page.goto('/chat');
    await showPoolEditor(page);

    const rows = page.locator('.ab-pool-agent-row');
    await rows.nth(0).locator('input[type="checkbox"]').check({ force: true });
    await rows.nth(1).locator('input[type="checkbox"]').check({ force: true });

    await expect(page.locator('#ab-pool-message')).toContainText('Champion');
  });
});

// =============================================================================
// Pool editor -- champion toggle
// =============================================================================

test.describe('A/B Pool Editor -- Champion Selection', () => {

  test('clicking champion button marks agent as champion', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminInactiveMocks(page);
    await page.goto('/chat');
    await showPoolEditor(page);

    const rows = page.locator('.ab-pool-agent-row');
    await rows.nth(0).locator('input[type="checkbox"]').check({ force: true });
    await rows.nth(1).locator('input[type="checkbox"]').check({ force: true });
    await rows.nth(0).locator('.ab-pool-champion-btn').click();

    await expect(rows.nth(0)).toHaveClass(/champion/);
  });

  test('switching champion removes previous champion', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');
    await showPoolEditor(page);

    const rows = page.locator('.ab-pool-agent-row');
    await rows.nth(2).locator('input[type="checkbox"]').check({ force: true });
    await rows.nth(2).locator('.ab-pool-champion-btn').click();

    await expect(page.locator('.ab-pool-agent-row.champion')).toHaveCount(1);
    await expect(rows.nth(2)).toHaveClass(/champion/);
  });

  test('unchecking champion removes champion status', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);
    await page.goto('/chat');
    await showPoolEditor(page);

    const championRow = page.locator('.ab-pool-agent-row.champion');
    await championRow.locator('input[type="checkbox"]').uncheck({ force: true });

    await expect(page.locator('.ab-pool-agent-row.champion')).toHaveCount(0);
  });
});

// =============================================================================
// A/B comparison streaming
// =============================================================================

test.describe('A/B Comparison Streaming', () => {

  test('sends A/B comparison and shows two arms', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    const abStream = createABStreamResponse({
      armAContent: 'Champion says hello',
      armBContent: 'Challenger says hi',
    });

    await page.route('**/api/ab/compare', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: abStream });
    });

    await page.goto('/chat');

    await page.getByLabel('Message input').fill('Hello');
    await page.getByRole('button', { name: 'Send message' }).click();

    const comparison = page.locator('#ab-comparison-active');
    await expect(comparison).toBeVisible();

    const arms = comparison.locator('.ab-arm');
    await expect(arms).toHaveCount(2);

    await expect(comparison.locator('.ab-arm-label').first()).toHaveText('Response A');
    await expect(comparison.locator('.ab-arm-label').nth(1)).toHaveText('Response B');
  });

  test('A/B stream populates content in both arms', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    const abStream = createABStreamResponse({
      armAContent: 'Alpha answer',
      armBContent: 'Beta answer',
    });

    await page.route('**/api/ab/compare', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: abStream });
    });

    await page.goto('/chat');

    await page.getByLabel('Message input').fill('Test AB');
    await page.getByRole('button', { name: 'Send message' }).click();

    const armA = page.locator('.ab-arm').first().locator('.message-content');
    const armB = page.locator('.ab-arm').nth(1).locator('.message-content');
    await expect(armA).toContainText('Alpha answer');
    await expect(armB).toContainText('Beta answer');
  });

  test('vote buttons appear after A/B stream completes', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    const abStream = createABStreamResponse();

    await page.route('**/api/ab/compare', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: abStream });
    });

    await page.goto('/chat');

    await page.getByLabel('Message input').fill('Test vote');
    await page.getByRole('button', { name: 'Send message' }).click();

    const voteContainer = page.locator('.ab-vote-container');
    await expect(voteContainer).toBeVisible();

    await expect(page.locator('.ab-vote-btn-a')).toBeVisible();
    await expect(page.locator('.ab-vote-btn-tie')).toBeVisible();
    await expect(page.locator('.ab-vote-btn-b')).toBeVisible();

    await expect(page.locator('.ab-vote-prompt')).toContainText('Which response do you prefer?');
  });

  test('input stays disabled until vote is submitted', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    const abStream = createABStreamResponse();

    await page.route('**/api/ab/compare', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: abStream });
    });

    await page.route('**/api/ab/preference', async (route) => {
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');

    await page.getByLabel('Message input').fill('Test disabled');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-vote-container')).toBeVisible();
    await expect(page.getByLabel('Message input')).toBeDisabled();

    await page.locator('.ab-vote-btn-a').click();

    await expect(page.getByLabel('Message input')).not.toBeDisabled();
  });
});

// =============================================================================
// Vote submission
// =============================================================================

test.describe('A/B Vote Submission', () => {

  async function setupABWithVote(page: any) {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    const abStream = createABStreamResponse({ comparisonId: 99 });

    await page.route('**/api/ab/compare', async (route: any) => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: abStream });
    });
  }

  test('voting A sends preference "a" to server', async ({ page }) => {
    await setupABWithVote(page);

    let submittedPreference: string | null = null;
    await page.route('**/api/ab/preference', async (route: any) => {
      const body = route.request().postDataJSON();
      submittedPreference = body.preference;
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');
    await page.getByLabel('Message input').fill('Vote A');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-vote-container')).toBeVisible();
    await page.locator('.ab-vote-btn-a').click();

    expect(submittedPreference).toBe('a');
  });

  test('voting B sends preference "b" to server', async ({ page }) => {
    await setupABWithVote(page);

    let submittedPreference: string | null = null;
    await page.route('**/api/ab/preference', async (route: any) => {
      const body = route.request().postDataJSON();
      submittedPreference = body.preference;
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');
    await page.getByLabel('Message input').fill('Vote B');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-vote-container')).toBeVisible();
    await page.locator('.ab-vote-btn-b').click();

    expect(submittedPreference).toBe('b');
  });

  test('voting Tie sends preference "tie" to server', async ({ page }) => {
    await setupABWithVote(page);

    let submittedPreference: string | null = null;
    await page.route('**/api/ab/preference', async (route: any) => {
      const body = route.request().postDataJSON();
      submittedPreference = body.preference;
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');
    await page.getByLabel('Message input').fill('Vote Tie');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-vote-container')).toBeVisible();
    await page.locator('.ab-vote-btn-tie').click();

    expect(submittedPreference).toBe('tie');
  });

  test('vote sends correct comparison_id', async ({ page }) => {
    await setupABWithVote(page);

    let sentComparisonId: number | null = null;
    await page.route('**/api/ab/preference', async (route: any) => {
      const body = route.request().postDataJSON();
      sentComparisonId = body.comparison_id;
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');
    await page.getByLabel('Message input').fill('Check ID');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-vote-container')).toBeVisible();
    await page.locator('.ab-vote-btn-a').click();

    expect(sentComparisonId).toBe(99);
  });

  test('vote buttons disappear after voting', async ({ page }) => {
    await setupABWithVote(page);

    await page.route('**/api/ab/preference', async (route: any) => {
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');
    await page.getByLabel('Message input').fill('Dismiss vote');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-vote-container')).toBeVisible();
    await page.locator('.ab-vote-btn-a').click();

    await expect(page.locator('.ab-vote-container')).toHaveCount(0);
  });

  test('choosing A collapses comparison to single message', async ({ page }) => {
    await setupABWithVote(page);

    await page.route('**/api/ab/preference', async (route: any) => {
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');
    await page.getByLabel('Message input').fill('Collapse test');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-vote-container')).toBeVisible();
    await page.locator('.ab-vote-btn-a').click();

    await expect(page.locator('#ab-comparison-active')).toHaveCount(0);
  });

  test('choosing Tie keeps both arms with tie styling', async ({ page }) => {
    await setupABWithVote(page);

    await page.route('**/api/ab/preference', async (route: any) => {
      await route.fulfill({ status: 200, json: { success: true } });
    });

    await page.goto('/chat');
    await page.getByLabel('Message input').fill('Tie test');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-vote-container')).toBeVisible();
    await page.locator('.ab-vote-btn-tie').click();

    await expect(page.locator('.ab-arm-tie')).toHaveCount(2);
  });
});

// =============================================================================
// A/B error handling
// =============================================================================

test.describe('A/B Error Handling', () => {

  test('error in A/B stream shows error message', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    const errorStream = JSON.stringify({
      type: 'error',
      message: 'Both arms timed out',
    }) + '\n';

    await page.route('**/api/ab/compare', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: errorStream });
    });

    await page.goto('/chat');

    await page.getByLabel('Message input').fill('Error test');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.locator('.ab-error-message')).toBeVisible();
    await expect(page.locator('.ab-error-message')).toContainText('Both arms timed out');
  });

  test('HTTP error from A/B compare re-enables input', async ({ page }) => {
    await setupBasicMocks(page);
    await setupABAdminMocks(page);

    await page.route('**/api/ab/compare', async (route) => {
      await route.fulfill({ status: 500, body: 'Internal Server Error' });
    });

    await page.goto('/chat');

    await page.getByLabel('Message input').fill('500 error');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.getByLabel('Message input')).not.toBeDisabled();
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
  });
});

// =============================================================================
// Normal mode -- A/B not engaged when pool is inactive
// =============================================================================

test.describe('A/B Inactive -- Normal Chat', () => {

  test('chat uses single stream when A/B pool is not enabled', async ({ page }) => {
    await setupBasicMocks(page);

    let abCompareCalled = false;
    await page.route('**/api/ab/compare', async (route) => {
      abCompareCalled = true;
      await route.fulfill({ status: 200, body: '' });
    });

    await page.route('**/api/get_chat_response_stream', async (route) => {
      const body = JSON.stringify({
        type: 'final',
        response: 'Normal response',
        message_id: 1,
        user_message_id: 1,
        conversation_id: 1,
      }) + '\n';
      await route.fulfill({ status: 200, contentType: 'text/plain', body });
    });

    await page.goto('/chat');

    await page.getByLabel('Message input').fill('Hello');
    await page.getByRole('button', { name: 'Send message' }).click();

    await page.waitForTimeout(500);
    expect(abCompareCalled).toBe(false);

    await expect(page.locator('#ab-comparison-active')).toHaveCount(0);
  });
});
