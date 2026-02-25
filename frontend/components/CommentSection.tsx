"use client";

import { useState, useEffect } from 'react';

interface Reply {
  id: string;
  author: string;
  author_name: string;
  text: string;
  created_at: string;
}

interface Comment {
  id: string;
  author: string;
  author_name: string;
  text: string;
  created_at: string;
  replies: Reply[];
}

export default function CommentSection({ docHash }: { docHash: string }) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [newComment, setNewComment] = useState("");
  const [replyText, setReplyText] = useState<{ [key: string]: string }>({});
  const [activeReplyId, setActiveReplyId] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<string>("");

  useEffect(() => {
    fetchComments();
    fetchCurrentUser();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docHash]);

  const fetchCurrentUser = async () => {
    const token = localStorage.getItem("token");
    if (!token) return;
    try {
      const res = await fetch("http://localhost:8000/auth/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setCurrentUser(data.username);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchComments = async () => {
    const token = localStorage.getItem("token");
    if (!token || !docHash) return;
    try {
      const res = await fetch(`http://localhost:8000/api/v1/comments/${docHash}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setComments(data);
      }
    } catch (e) {
      console.error("Failed to fetch comments", e);
    }
  };

  const handleSubmitComment = async () => {
    if (!newComment.trim()) return;
    const token = localStorage.getItem("token");
    if (!token) return;

    try {
      const res = await fetch("http://localhost:8000/api/v1/comments", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ doc_hash: docHash, text: newComment }),
      });
      if (res.ok) {
        setNewComment("");
        fetchComments();
      } else {
        const err = await res.json();
        alert(err.detail || "댓글 작성 실패");
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleSubmitReply = async (commentId: string) => {
    const text = replyText[commentId];
    if (!text?.trim()) return;
    const token = localStorage.getItem("token");
    if (!token) return;

    try {
      const res = await fetch(`http://localhost:8000/api/v1/comments/${commentId}/replies`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ text }),
      });
      if (res.ok) {
        setReplyText({ ...replyText, [commentId]: "" });
        setActiveReplyId(null);
        fetchComments();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleDelete = async (id: string, type: 'comments' | 'replies') => {
    if (!confirm("정말 삭제하시겠습니까?")) return;
    const token = localStorage.getItem("token");
    if (!token) return;

    try {
      const res = await fetch(`http://localhost:8000/api/v1/${type}/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        fetchComments();
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mt-8">
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
        💬 팀 의견 공유 <span className="text-sm font-normal text-gray-500">({comments.length})</span>
      </h3>
      
      <div className="space-y-6 mb-8">
        {comments.length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400 text-sm text-center py-4">아직 작성된 의견이 없습니다.</p>
        ) : (
          comments.map((comment) => (
            <div key={comment.id} className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center text-indigo-600 dark:text-indigo-400 font-bold text-xs shrink-0">
                {comment.author_name[0]}
              </div>
              <div className="flex-1">
                <div className="bg-gray-50 dark:bg-gray-700/30 rounded-lg p-3">
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-semibold text-sm text-gray-900 dark:text-white">{comment.author_name}</span>
                    <span className="text-xs text-gray-500">{new Date(comment.created_at).toLocaleString()}</span>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{comment.text}</p>
                </div>
                <div className="flex gap-3 mt-1 ml-1 text-xs">
                  <button onClick={() => setActiveReplyId(activeReplyId === comment.id ? null : comment.id)} className="text-gray-500 hover:text-indigo-600 font-medium">답글 달기</button>
                  {comment.author === currentUser && (
                    <button onClick={() => handleDelete(comment.id, 'comments')} className="text-gray-500 hover:text-red-600">삭제</button>
                  )}
                </div>

                {/* Replies */}
                {comment.replies.map((reply) => (
                  <div key={reply.id} className="flex gap-3 mt-3 ml-4">
                    <div className="w-6 h-6 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center text-gray-600 dark:text-gray-400 font-bold text-[10px] shrink-0">
                      {reply.author_name[0]}
                    </div>
                    <div className="flex-1">
                      <div className="bg-gray-50 dark:bg-gray-700/30 rounded-lg p-2 px-3">
                        <div className="flex justify-between items-start mb-1">
                          <span className="font-semibold text-xs text-gray-900 dark:text-white">{reply.author_name}</span>
                          <span className="text-[10px] text-gray-500">{new Date(reply.created_at).toLocaleString()}</span>
                        </div>
                        <p className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{reply.text}</p>
                      </div>
                      {reply.author === currentUser && (
                        <button onClick={() => handleDelete(reply.id, 'replies')} className="text-xs text-gray-500 hover:text-red-600 mt-1 ml-1">삭제</button>
                      )}
                    </div>
                  </div>
                ))}

                {activeReplyId === comment.id && (
                  <div className="mt-3 ml-4 flex gap-2">
                    <input
                      type="text"
                      value={replyText[comment.id] || ""}
                      onChange={(e) => setReplyText({ ...replyText, [comment.id]: e.target.value })}
                      placeholder="답글을 입력하세요..."
                      className="flex-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-indigo-500"
                      onKeyDown={(e) => e.key === 'Enter' && handleSubmitReply(comment.id)}
                    />
                    <button onClick={() => handleSubmitReply(comment.id)} className="bg-indigo-600 text-white text-xs px-3 py-2 rounded-lg hover:bg-indigo-700">등록</button>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-500 dark:text-gray-400 text-xs shrink-0">
          나
        </div>
        <div className="flex-1">
          <textarea
            value={newComment}
            onChange={(e) => setNewComment(e.target.value)}
            placeholder="이 문서에 대한 의견을 남겨주세요..."
            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-3 text-sm dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-indigo-500 min-h-[80px]"
          />
          <div className="flex justify-end mt-2">
            <button 
              onClick={handleSubmitComment}
              disabled={!newComment.trim()}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              의견 등록
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
