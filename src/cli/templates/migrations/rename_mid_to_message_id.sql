-- Migration: Rename 'mid' columns to 'message_id' for consistency
-- Issue: https://github.com/archi-physics/archi/issues/343
--
-- Run this script against existing deployments to update the schema.
-- New deployments using init.sql already have the correct column names.

-- feedback table: rename 'mid' -> 'message_id'
ALTER TABLE feedback RENAME COLUMN mid TO message_id;
ALTER INDEX IF EXISTS idx_feedback_mid RENAME TO idx_feedback_message_id;

-- timing table: rename 'mid' -> 'message_id'
ALTER TABLE timing RENAME COLUMN mid TO message_id;

-- ab_comparisons table: rename the three *_mid columns
ALTER TABLE ab_comparisons RENAME COLUMN user_prompt_mid TO user_prompt_message_id;
ALTER TABLE ab_comparisons RENAME COLUMN response_a_mid TO response_a_message_id;
ALTER TABLE ab_comparisons RENAME COLUMN response_b_mid TO response_b_message_id;
