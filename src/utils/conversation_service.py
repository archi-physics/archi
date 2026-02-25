"""
ConversationService - Conversation tracking with model/pipeline info.

This service manages conversation storage with the PostgreSQL-consolidated schema,
storing model_used and pipeline_used directly instead of config foreign keys.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import psycopg2
from psycopg2.extras import execute_values

from src.utils.sql import (
    SQL_INSERT_CONVO,
    SQL_INSERT_AB_COMPARISON,
    SQL_UPDATE_AB_PREFERENCE,
    SQL_GET_AB_COMPARISON,
    SQL_GET_PENDING_AB_COMPARISON,
    SQL_DELETE_AB_COMPARISON,
    SQL_GET_AB_COMPARISONS_BY_CONVERSATION,
)


@dataclass
class Message:
    """A conversation message."""
    message_id: Optional[int] = None
    conversation_id: str = ""
    sender: str = ""  # 'user' or 'assistant'
    content: str = ""
    link: Optional[str] = None
    context: Optional[str] = None  # JSON string of context/sources
    ts: Optional[datetime] = None
    model_used: Optional[str] = None
    pipeline_used: Optional[str] = None
    archi_service: str = "chat"


@dataclass
class ABComparison:
    """An A/B comparison between two model responses."""
    comparison_id: Optional[int] = None
    conversation_id: str = ""
    user_prompt_mid: int = 0  # message_id of user prompt
    response_a_mid: int = 0   # message_id of response A
    response_b_mid: int = 0   # message_id of response B
    model_a: str = ""
    pipeline_a: str = ""
    model_b: str = ""
    pipeline_b: str = ""
    is_config_a_first: bool = True  # Which response shown first in UI
    preference: Optional[str] = None  # 'a', 'b', 'tie', or None
    preference_ts: Optional[datetime] = None
    created_at: Optional[datetime] = None


class ConversationService:
    """
    Service for managing conversation tracking with model/pipeline info.
    
    Supports:
    - Storing messages with model_used/pipeline_used (no config FK)
    - A/B comparisons with model_a/model_b instead of config_id
    - Batch message insertion for efficiency
    - Conversation history retrieval
    """
    
    def __init__(self, connection_pool=None, connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize ConversationService.
        
        Args:
            connection_pool: ConnectionPool instance (preferred)
            connection_params: Direct connection parameters (fallback)
        """
        self._pool = connection_pool
        self._conn_params = connection_params
    
    def _get_connection(self):
        """Get a database connection."""
        if self._pool:
            return self._pool.get_connection()
        elif self._conn_params:
            return psycopg2.connect(**self._conn_params)
        else:
            raise ValueError("No connection pool or params provided")
    
    def _release_connection(self, conn):
        """Release connection back to pool."""
        if self._pool:
            self._pool.release_connection(conn)
        else:
            conn.close()
    
    # =========================================================================
    # Message Operations
    # =========================================================================
    
    def insert_message(self, message: Message) -> int:
        """
        Insert a single message.
        
        Args:
            message: Message to insert
            
        Returns:
            message_id of inserted message
        """
        return self.insert_messages([message])[0]
    
    def insert_messages(self, messages: List[Message]) -> List[int]:
        """
        Insert multiple messages in a batch.
        
        Args:
            messages: List of messages to insert
            
        Returns:
            List of message_ids
        """
        if not messages:
            return []
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                values = [
                    (
                        m.archi_service,
                        m.conversation_id,
                        m.sender,
                        m.content,
                        m.link or "",  # DB requires NOT NULL
                        m.context or "",  # DB requires NOT NULL
                        m.ts or datetime.now(),
                        m.model_used,
                        m.pipeline_used,
                    )
                    for m in messages
                ]
                result = execute_values(
                    cur,
                    SQL_INSERT_CONVO,
                    values,
                    fetch=True
                )
                conn.commit()
                return [row[0] for row in result]
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)
    
    def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Message]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation to retrieve
            limit: Max messages to return
            offset: Pagination offset
            
        Returns:
            List of Messages ordered by timestamp
        """
        query = """
            SELECT message_id, conversation_id, sender, content, link, context, 
                   ts, model_used, pipeline_used, archi_service
            FROM conversations
            WHERE conversation_id = %s
            ORDER BY ts ASC
            LIMIT %s OFFSET %s;
        """
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (conversation_id, limit, offset))
                rows = cur.fetchall()
                
                return [
                    Message(
                        message_id=row[0],
                        conversation_id=row[1],
                        sender=row[2],
                        content=row[3],
                        link=row[4],
                        context=row[5],
                        ts=row[6],
                        model_used=row[7],
                        pipeline_used=row[8],
                        archi_service=row[9],
                    )
                    for row in rows
                ]
        finally:
            self._release_connection(conn)
    
    def get_user_conversations(
        self,
        user_id: str,
        archi_service: str = "chat",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get list of conversations for a user.
        
        Args:
            user_id: User identifier (stored in conversation_id prefix)
            archi_service: Service filter
            limit: Max conversations to return
            
        Returns:
            List of conversation summaries with last message info
        """
        # Assuming conversation_id format: "user_<user_id>_<uuid>"
        query = """
            SELECT DISTINCT ON (conversation_id)
                conversation_id,
                MAX(ts) as last_message_at,
                COUNT(*) as message_count
            FROM conversations
            WHERE conversation_id LIKE %s
              AND archi_service = %s
            GROUP BY conversation_id
            ORDER BY conversation_id, last_message_at DESC
            LIMIT %s;
        """
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (f"user_{user_id}_%", archi_service, limit))
                rows = cur.fetchall()
                
                return [
                    {
                        "conversation_id": row[0],
                        "last_message_at": row[1],
                        "message_count": row[2],
                    }
                    for row in rows
                ]
        finally:
            self._release_connection(conn)
    
    # =========================================================================
    # A/B Comparison Operations
    # =========================================================================
    
    def create_ab_comparison(
        self,
        conversation_id: str,
        user_prompt_mid: int,
        response_a_mid: int,
        response_b_mid: int,
        model_a: str,
        pipeline_a: str,
        model_b: str,
        pipeline_b: str,
        is_config_a_first: bool = True,
    ) -> int:
        """
        Create a new A/B comparison.
        
        Args:
            conversation_id: Conversation this comparison belongs to
            user_prompt_mid: Message ID of user prompt
            response_a_mid: Message ID of response A
            response_b_mid: Message ID of response B
            model_a: Model identifier for response A
            pipeline_a: Pipeline identifier for response A
            model_b: Model identifier for response B
            pipeline_b: Pipeline identifier for response B
            is_config_a_first: Whether response A shown first
            
        Returns:
            comparison_id
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    SQL_INSERT_AB_COMPARISON,
                    (
                        conversation_id,
                        user_prompt_mid,
                        response_a_mid,
                        response_b_mid,
                        model_a,
                        pipeline_a,
                        model_b,
                        pipeline_b,
                        is_config_a_first,
                    )
                )
                comparison_id = cur.fetchone()[0]
                conn.commit()
                return comparison_id
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)
    
    def record_ab_preference(
        self,
        comparison_id: int,
        preference: str,
    ) -> None:
        """
        Record user's preference for an A/B comparison.
        
        Args:
            comparison_id: ID of comparison
            preference: 'a', 'b', 'tie', or 'skip'
        """
        if preference not in ('a', 'b', 'tie', 'skip'):
            raise ValueError(f"Invalid preference: {preference}")
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    SQL_UPDATE_AB_PREFERENCE,
                    (preference, datetime.now(), comparison_id)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)
    
    def get_ab_comparison(self, comparison_id: int) -> Optional[ABComparison]:
        """
        Get a specific A/B comparison.
        
        Args:
            comparison_id: ID of comparison
            
        Returns:
            ABComparison or None if not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(SQL_GET_AB_COMPARISON, (comparison_id,))
                row = cur.fetchone()
                
                if not row:
                    return None
                
                return ABComparison(
                    comparison_id=row[0],
                    conversation_id=row[1],
                    user_prompt_mid=row[2],
                    response_a_mid=row[3],
                    response_b_mid=row[4],
                    model_a=row[5],
                    pipeline_a=row[6],
                    model_b=row[7],
                    pipeline_b=row[8],
                    is_config_a_first=row[9],
                    preference=row[10],
                    preference_ts=row[11],
                    created_at=row[12],
                )
        finally:
            self._release_connection(conn)
    
    def get_pending_ab_comparison(
        self,
        conversation_id: str,
    ) -> Optional[ABComparison]:
        """
        Get the most recent pending (unvoted) A/B comparison.
        
        Args:
            conversation_id: Conversation to check
            
        Returns:
            ABComparison or None if no pending comparisons
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(SQL_GET_PENDING_AB_COMPARISON, (conversation_id,))
                row = cur.fetchone()
                
                if not row:
                    return None
                
                return ABComparison(
                    comparison_id=row[0],
                    conversation_id=row[1],
                    user_prompt_mid=row[2],
                    response_a_mid=row[3],
                    response_b_mid=row[4],
                    model_a=row[5],
                    pipeline_a=row[6],
                    model_b=row[7],
                    pipeline_b=row[8],
                    is_config_a_first=row[9],
                    preference=row[10],
                    preference_ts=row[11],
                    created_at=row[12],
                )
        finally:
            self._release_connection(conn)
    
    def get_conversation_ab_comparisons(
        self,
        conversation_id: str,
    ) -> List[ABComparison]:
        """
        Get all A/B comparisons for a conversation.
        
        Args:
            conversation_id: Conversation to retrieve comparisons for
            
        Returns:
            List of ABComparisons ordered by creation time
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    SQL_GET_AB_COMPARISONS_BY_CONVERSATION,
                    (conversation_id,)
                )
                rows = cur.fetchall()
                
                return [
                    ABComparison(
                        comparison_id=row[0],
                        conversation_id=row[1],
                        user_prompt_mid=row[2],
                        response_a_mid=row[3],
                        response_b_mid=row[4],
                        model_a=row[5],
                        pipeline_a=row[6],
                        model_b=row[7],
                        pipeline_b=row[8],
                        is_config_a_first=row[9],
                        preference=row[10],
                        preference_ts=row[11],
                        created_at=row[12],
                    )
                    for row in rows
                ]
        finally:
            self._release_connection(conn)
    
    def delete_ab_comparison(self, comparison_id: int) -> bool:
        """
        Delete an A/B comparison.
        
        Args:
            comparison_id: ID of comparison to delete
            
        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(SQL_DELETE_AB_COMPARISON, (comparison_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)
    
    # =========================================================================
    # Analytics Queries
    # =========================================================================
    
    def get_model_comparison_stats(
        self,
        model_a: Optional[str] = None,
        model_b: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregated A/B comparison statistics.
        
        Args:
            model_a: Filter by model A (optional)
            model_b: Filter by model B (optional)
            start_date: Start of time range (optional)
            end_date: End of time range (optional)
            
        Returns:
            Dict with statistics: total, preference_counts, win_rates
        """
        query = """
            SELECT 
                model_a, model_b,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE preference = 'a') as wins_a,
                COUNT(*) FILTER (WHERE preference = 'b') as wins_b,
                COUNT(*) FILTER (WHERE preference = 'tie') as ties,
                COUNT(*) FILTER (WHERE preference = 'skip') as skips,
                COUNT(*) FILTER (WHERE preference IS NULL) as pending
            FROM ab_comparisons
            WHERE 1=1
        """
        params = []
        
        if model_a:
            query += " AND model_a = %s"
            params.append(model_a)
        if model_b:
            query += " AND model_b = %s"
            params.append(model_b)
        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)
        
        query += " GROUP BY model_a, model_b"
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    total_voted = row[3] + row[4] + row[5]  # wins_a + wins_b + ties
                    results.append({
                        "model_a": row[0],
                        "model_b": row[1],
                        "total": row[2],
                        "wins_a": row[3],
                        "wins_b": row[4],
                        "ties": row[5],
                        "skips": row[6],
                        "pending": row[7],
                        "win_rate_a": row[3] / total_voted if total_voted > 0 else 0,
                        "win_rate_b": row[4] / total_voted if total_voted > 0 else 0,
                    })
                
                return {"comparisons": results}
        finally:
            self._release_connection(conn)
    
    def get_model_usage_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        archi_service: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get model usage statistics from conversations.
        
        Args:
            start_date: Start of time range (optional)
            end_date: End of time range (optional)
            archi_service: Service filter (optional)
            
        Returns:
            List of model usage stats
        """
        query = """
            SELECT 
                model_used,
                pipeline_used,
                COUNT(*) as message_count,
                COUNT(DISTINCT conversation_id) as conversation_count
            FROM conversations
            WHERE sender = 'assistant'
              AND model_used IS NOT NULL
        """
        params = []
        
        if start_date:
            query += " AND ts >= %s"
            params.append(start_date)
        if end_date:
            query += " AND ts <= %s"
            params.append(end_date)
        if archi_service:
            query += " AND archi_service = %s"
            params.append(archi_service)
        
        query += " GROUP BY model_used, pipeline_used ORDER BY message_count DESC"
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                
                return [
                    {
                        "model": row[0],
                        "pipeline": row[1],
                        "message_count": row[2],
                        "conversation_count": row[3],
                    }
                    for row in rows
                ]
        finally:
            self._release_connection(conn)
