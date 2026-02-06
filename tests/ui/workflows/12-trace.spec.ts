/**
 * Workflow 12: Agent Trace/Tool Call Visualization Tests
 * 
 * Tests for viewing tool calls in message responses.
 * The current UI may show tool calls inline or in collapsible sections.
 */
import { test, expect, setupBasicMocks, createToolCallEvents } from '../fixtures';

test.describe('Agent Trace Visualization', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  test('response with tool calls displays correctly', async ({ page }) => {
    const events = createToolCallEvents('search', { query: 'test' }, 'Found results');
    
    await page.route('**/api/get_chat_response_stream', async (route) => {
      let body = events.map(e => JSON.stringify(e)).join('\n') + '\n';
      body += '{"type":"final","response":"Based on the search, here is the answer.","message_id":1,"user_message_id":1,"conversation_id":1}\n';
      
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body,
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Search for info');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Should show the final response
    await expect(page.getByText('Based on the search')).toBeVisible({ timeout: 5000 });
  });

  test('streaming response shows content progressively', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      const chunks = [
        '{"type":"chunk","content":"First part. "}',
        '{"type":"chunk","content":"Second part. "}',
        '{"type":"chunk","content":"Third part."}',
        '{"type":"done"}',
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: chunks.join('\n') + '\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Should show the full response
    await expect(page.getByText('First part')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('Third part')).toBeVisible();
  });

  test('tool call events do not break response flow', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      const events = [
        '{"type":"tool_call","name":"search","args":{"query":"test"}}',
        '{"type":"tool_result","result":"found data"}',
        '{"type":"chunk","content":"Here is the answer."}',
        '{"type":"done"}',
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: events.join('\n') + '\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Search');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // The response should be visible
    await expect(page.getByText('Here is the answer')).toBeVisible({ timeout: 5000 });
  });

  test('message meta shows agent info after response', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"chunk","content":"Response"}\n{"type":"done"}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Should show agent name in the message area (not header/sidebar)
    await expect(page.getByRole('main').getByText('archi').first()).toBeVisible({ timeout: 5000 });
  });

  test('agent activity panel shows with tool calls', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      const events = [
        '{"type":"tool_start","tool_call_id":"tc_1","tool_name":"search","tool_args":{"query":"test"}}',
        '{"type":"tool_output","tool_call_id":"tc_1","output":"Found results"}',
        '{"type":"tool_end","tool_call_id":"tc_1","status":"success","duration_ms":150}',
        '{"type":"final","response":"Here is the answer.","message_id":1,"user_message_id":1,"conversation_id":1}',
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: events.join('\n') + '\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Search');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Should show Agent Activity panel
    await expect(page.getByText('Agent Activity')).toBeVisible({ timeout: 5000 });
  });

  test('thinking step displays during agent reasoning', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      const events = [
        '{"type":"thinking_start","step_id":"think_1"}',
        '{"type":"thinking_end","step_id":"think_1","duration_ms":500,"thinking_content":"Let me analyze this..."}',
        '{"type":"final","response":"Here is my answer.","message_id":1,"user_message_id":1,"conversation_id":1}',
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: events.join('\n') + '\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Think about this');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Should show Agent Activity with thinking
    await expect(page.getByText('Agent Activity')).toBeVisible({ timeout: 5000 });
    // Expand the trace to see thinking (click toggle button)
    await page.locator('.trace-toggle').click();
    // Wait for container to expand (collapsed class removed)
    await expect(page.locator('.trace-container:not(.collapsed)')).toBeVisible({ timeout: 2000 });
    await expect(page.locator('.thinking-step .step-label')).toBeVisible();
  });

  test('token usage displays in context meter', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      const events = [
        '{"type":"thinking_start","step_id":"think_1"}',
        '{"type":"thinking_end","step_id":"think_1","duration_ms":100,"thinking_content":"OK"}',
        '{"type":"final","response":"Here is the answer.","message_id":1,"user_message_id":1,"conversation_id":1,"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}}',
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: events.join('\n') + '\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Expand trace to see context meter
    await expect(page.getByText('Agent Activity')).toBeVisible({ timeout: 5000 });
    await page.locator('.trace-toggle').click();
    await expect(page.locator('.trace-container:not(.collapsed)')).toBeVisible({ timeout: 2000 });
    
    // Should show token usage in the meter label
    await expect(page.locator('.meter-label')).toBeVisible();
  });

  test('elapsed timer shows during streaming', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      // Delay slightly to show timer in action
      await new Promise(resolve => setTimeout(resolve, 200));
      const events = [
        '{"type":"thinking_start","step_id":"think_1"}',
        '{"type":"thinking_end","step_id":"think_1","duration_ms":100}',
        '{"type":"final","response":"Done.","message_id":1,"user_message_id":1,"conversation_id":1}',
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: events.join('\n') + '\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Should show Agent Activity panel with timer
    await expect(page.getByText('Agent Activity')).toBeVisible({ timeout: 5000 });
    // Timer should have some time displayed
    await expect(page.locator('.trace-timer')).toBeVisible();
  });

  test('tool step is expandable to show args and output', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      const events = [
        '{"type":"tool_start","tool_call_id":"tc_1","tool_name":"search","tool_args":{"query":"test query"}}',
        '{"type":"tool_output","tool_call_id":"tc_1","output":"Search results here"}',
        '{"type":"tool_end","tool_call_id":"tc_1","status":"success","duration_ms":150}',
        '{"type":"final","response":"Here is the answer.","message_id":1,"user_message_id":1,"conversation_id":1}',
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: events.join('\n') + '\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Search');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Expand trace panel
    await expect(page.getByText('Agent Activity')).toBeVisible({ timeout: 5000 });
    await page.locator('.trace-toggle').click();
    await expect(page.locator('.trace-container:not(.collapsed)')).toBeVisible({ timeout: 2000 });
    
    // Click on tool step header to expand
    await expect(page.locator('.tool-step .step-header')).toBeVisible();
    await page.locator('.tool-step .step-header').click();
    
    // Wait for step details to be visible
    await expect(page.locator('.tool-step .step-details')).toBeVisible({ timeout: 2000 });
    await expect(page.locator('.tool-step').getByText('Arguments')).toBeVisible();
  });
});

test.describe('Feedback Buttons', () => {
  test.beforeEach(async ({ page }) => {
    await setupBasicMocks(page);
  });

  test('feedback buttons appear on hover for assistant messages', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Here is the answer.","message_id":1,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Wait for response
    await expect(page.getByText('Here is the answer')).toBeVisible({ timeout: 5000 });
    
    // Hover over the assistant message
    await page.locator('.message.assistant').hover();
    
    // Should show feedback buttons with proper labels
    await expect(page.locator('.message-actions')).toBeVisible();
    await expect(page.locator('.feedback-like')).toBeVisible();
    await expect(page.locator('.feedback-dislike')).toBeVisible();
    await expect(page.locator('.feedback-comment')).toBeVisible();
  });

  test('like button sends feedback and shows active state', async ({ page }) => {
    let likeCallCount = 0;
    
    await page.route('**/api/like', async (route) => {
      likeCallCount++;
      // First call: set like state
      // Second call: toggle off (would return null state)
      const state = likeCallCount === 1 ? 'like' : null;
      await route.fulfill({ 
        status: 200, 
        json: { message: state ? 'Liked' : 'Reaction removed', state } 
      });
    });

    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Answer","message_id":123,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.getByText('Answer', { exact: true })).toBeVisible({ timeout: 5000 });
    
    // Click like button
    await page.locator('.message.assistant').hover();
    await page.locator('.feedback-like').click();
    
    // Verify like state is shown
    await expect(page.locator('.message-actions.feedback-like-active')).toBeVisible();
    expect(likeCallCount).toBe(1);
    
    // Click again to toggle off
    await page.locator('.feedback-like').click();
    await expect(page.locator('.message-actions.feedback-like-active')).not.toBeVisible();
    expect(likeCallCount).toBe(2);
  });

  test('switching from like to dislike updates state correctly', async ({ page }) => {
    await page.route('**/api/like', async (route) => {
      await route.fulfill({ 
        status: 200, 
        json: { message: 'Liked', state: 'like' } 
      });
    });

    await page.route('**/api/dislike', async (route) => {
      await route.fulfill({ 
        status: 200, 
        json: { message: 'Disliked', state: 'dislike' } 
      });
    });

    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Answer","message_id":123,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.getByText('Answer', { exact: true })).toBeVisible({ timeout: 5000 });
    
    // Like first
    await page.locator('.message.assistant').hover();
    await page.locator('.feedback-like').click();
    await expect(page.locator('.message-actions.feedback-like-active')).toBeVisible();
    
    // Then dislike - should switch
    await page.locator('.feedback-dislike').click();
    await expect(page.locator('.message-actions.feedback-dislike-active')).toBeVisible();
    await expect(page.locator('.message-actions.feedback-like-active')).not.toBeVisible();
  });

  test('comment button opens feedback modal', async ({ page }) => {
    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Answer","message_id":1,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.getByText('Answer', { exact: true })).toBeVisible({ timeout: 5000 });
    
    // Click comment button
    await page.locator('.message.assistant').hover();
    await page.locator('.feedback-comment').click();
    
    // Modal should be visible with proper content
    await expect(page.getByText('Send Feedback')).toBeVisible();
    await expect(page.getByText('Help us improve')).toBeVisible();
    await expect(page.locator('#feedback-text')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Submit Feedback' })).toBeVisible();
  });

  test('feedback modal submits text and closes', async ({ page }) => {
    let feedbackReceived = '';
    
    await page.route('**/api/text_feedback', async (route) => {
      const body = await route.request().postDataJSON();
      feedbackReceived = body.feedback_msg;
      await route.fulfill({ status: 200, json: { message: 'Feedback submitted' } });
    });

    await page.route('**/api/get_chat_response_stream', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '{"type":"final","response":"Answer","message_id":1,"user_message_id":1,"conversation_id":1}\n',
      });
    });

    await page.goto('/chat');
    
    await page.getByLabel('Message input').fill('Test');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    await expect(page.getByText('Answer', { exact: true })).toBeVisible({ timeout: 5000 });
    
    // Open modal and submit feedback
    await page.locator('.message.assistant').hover();
    await page.locator('.feedback-comment').click();
    
    await page.locator('#feedback-text').fill('This response was very helpful!');
    await page.getByRole('button', { name: 'Submit Feedback' }).click();
    
    // Modal should close
    await expect(page.locator('#feedback-modal')).not.toBeVisible();
    
    // Verify feedback was sent
    expect(feedbackReceived).toBe('This response was very helpful!');
  });
});
