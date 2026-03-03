"use client";

import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
  FormEvent,
  Fragment,
} from "react";
import { Button } from "@/components/ui/button";
import {
  Square,
  ArrowUp,
  CheckCircle,
  Clock,
  Circle,
  FileIcon,
} from "lucide-react";
import { ChatMessage } from "@/app/components/ChatMessage";
import type {
  TodoItem,
  ToolCall,
  ActionRequest,
  ReviewConfig,
} from "@/app/types/types";
import { Assistant, Message } from "@langchain/langgraph-sdk";
import { extractStringFromMessageContent } from "@/app/utils/utils";
import { useChatContext } from "@/providers/ChatProvider";
import { cn } from "@/lib/utils";
import { useStickToBottom } from "use-stick-to-bottom";
import { FilesPopover } from "@/app/components/TasksFilesSidebar";

interface ChatInterfaceProps {
  assistant: Assistant | null;
}

// 8 creative messages shown in the right panel after a 20-second delay.
// {todo} is replaced at runtime with the first pending todo's content.
const DELAY_MESSAGES = [
  `Still brewing results for "{todo}" ☕ — grab a coffee, this one's a deep thinker.`,
  `Your AI is fully lost in "{todo}" 🔍 — perfect time to stretch those legs!`,
  `Untangling "{todo}" thread by thread 🧵 — why not refill that water bottle?`,
  `"{todo}" has the AI in philosopher mode 🧠 — you've earned a snack break.`,
  `Wrestling with "{todo}" 🏋️ — the AI is giving it absolutely everything it's got.`,
  `Running heavy analysis on "{todo}" ⚙️ — patience is a virtue (coffee makes it easier).`,
  `Laser-focused on "{todo}" 🎯 — this is a good time to blink, you haven't in a while.`,
  `Deep in the weeds with "{todo}" 🌿 — pour something warm, we'll be right back.`,
  `"{todo}" demands serious brain power 💡 — a quick lap around the office might help!`,
  `The numbers behind "{todo}" don't lie, but they do take time 📊 — sit tight, we've got this.`,
];
const getStatusIcon = (status: TodoItem["status"], className?: string) => {
  switch (status) {
    case "completed":
      return (
        <CheckCircle
          size={16}
          className={cn("text-success/80", className)}
        />
      );
    case "in_progress":
      return (
        <Clock
          size={16}
          className={cn("text-warning/80", className)}
        />
      );
    default:
      return (
        <Circle
          size={16}
          className={cn("text-tertiary/70", className)}
        />
      );
  }
};

export const ChatInterface = React.memo<ChatInterfaceProps>(({ assistant }) => {
  const [metaOpen, setMetaOpen] = useState<"tasks" | "files" | null>(null);
  const tasksContainerRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const [input, setInput] = useState("");

  // ADDED: state for the right-panel delay message and its typing animation
  const [delayMessage, setDelayMessage] = useState<string | null>(null);
  const [typedText, setTypedText] = useState("");
  // Incremented each time the panel is dismissed by an arriving AI message
  // (not by isLoading going false). Adding it to the timer effect's deps
  // forces the effect to re-run and start a fresh 20-second countdown for
  // any subsequent delay in the same stream.
  const [timerEpoch, setTimerEpoch] = useState(0);
  const delayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const typingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Tracks whether the delay message is already on screen so we don't restart
  // the callback while the user is reading it.
  const delayShownRef = useRef(false);
  // Stores the message count at the moment the delay panel appeared, so we can
  // detect new AI messages arriving while isLoading is still true.
  const msgCountAtDelayRef = useRef(0);
  const { scrollRef, contentRef } = useStickToBottom();

  const {
    stream,
    messages,
    todos,
    files,
    ui,
    setFiles,
    isLoading,
    isThreadLoading,
    interrupt,
    sendMessage,
    stopStream,
    resumeInterrupt,
  } = useChatContext();

  const submitDisabled = isLoading || !assistant;

  const handleSubmit = useCallback(
    (e?: FormEvent) => {
      if (e) {
        e.preventDefault();
      }
      const messageText = input.trim();
      if (!messageText || isLoading || submitDisabled) return;
      sendMessage(messageText);
      setInput("");
    },
    [input, isLoading, sendMessage, setInput, submitDisabled]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (submitDisabled) return;
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit, submitDisabled]
  );

  // TODO: can we make this part of the hook?
  const processedMessages = useMemo(() => {
    /*
     1. Loop through all messages
     2. For each AI message, add the AI message, and any tool calls to the messageMap
     3. For each tool message, find the corresponding tool call in the messageMap and update the status and output
    */
    const messageMap = new Map<
      string,
      { message: Message; toolCalls: ToolCall[] }
    >();
    messages.forEach((message: Message) => {
      if (message.type === "ai") {
        const toolCallsInMessage: Array<{
          id?: string;
          function?: { name?: string; arguments?: unknown };
          name?: string;
          type?: string;
          args?: unknown;
          input?: unknown;
        }> = [];
        if (
          message.additional_kwargs?.tool_calls &&
          Array.isArray(message.additional_kwargs.tool_calls)
        ) {
          toolCallsInMessage.push(...message.additional_kwargs.tool_calls);
        } else if (message.tool_calls && Array.isArray(message.tool_calls)) {
          toolCallsInMessage.push(
            ...message.tool_calls.filter(
              (toolCall: { name?: string }) => toolCall.name !== ""
            )
          );
        } else if (Array.isArray(message.content)) {
          const toolUseBlocks = message.content.filter(
            (block: { type?: string }) => block.type === "tool_use"
          );
          toolCallsInMessage.push(...toolUseBlocks);
        }
        const toolCallsWithStatus = toolCallsInMessage.map(
          (toolCall: {
            id?: string;
            function?: { name?: string; arguments?: unknown };
            name?: string;
            type?: string;
            args?: unknown;
            input?: unknown;
          }) => {
            const name =
              toolCall.function?.name ||
              toolCall.name ||
              toolCall.type ||
              "unknown";
            const args =
              toolCall.function?.arguments ||
              toolCall.args ||
              toolCall.input ||
              {};
            return {
              id: toolCall.id || `tool-${Math.random()}`,
              name,
              args,
              status: interrupt ? "interrupted" : ("pending" as const),
            } as ToolCall;
          }
        );
        messageMap.set(message.id!, {
          message,
          toolCalls: toolCallsWithStatus,
        });
      } else if (message.type === "tool") {
        const toolCallId = message.tool_call_id;
        if (!toolCallId) {
          return;
        }
        for (const [, data] of messageMap.entries()) {
          const toolCallIndex = data.toolCalls.findIndex(
            (tc: ToolCall) => tc.id === toolCallId
          );
          if (toolCallIndex === -1) {
            continue;
          }
          data.toolCalls[toolCallIndex] = {
            ...data.toolCalls[toolCallIndex],
            status: "completed" as const,
            result: extractStringFromMessageContent(message),
          };
          break;
        }
      } else if (message.type === "human") {
        messageMap.set(message.id!, {
          message,
          toolCalls: [],
        });
      }
    });
    const processedArray = Array.from(messageMap.values());
    return processedArray.map((data, index) => {
      const prevMessage = index > 0 ? processedArray[index - 1].message : null;
      return {
        ...data,
        showAvatar: data.message.type !== prevMessage?.type,
      };
    });
  }, [messages, interrupt]);

  const groupedTodos = {
    in_progress: todos.filter((t) => t.status === "in_progress"),
    pending: todos.filter((t) => t.status === "pending"),
    completed: todos.filter((t) => t.status === "completed"),
  };

  const hasTasks = todos.length > 0;
  const hasFiles = Object.keys(files).length > 0;

  // ADDED: Walk stream.values.messages in reverse to find the most recent AI
  // message that carries reasoning_content in its additional_kwargs.
  // Returns null when no such message exists yet.
  // Recomputes only when the messages list reference changes (new streamed chunk).
  const latestReasoning = useMemo<string | null>(() => {
    const msgs = stream.values?.messages ?? [];
    for (let i = msgs.length - 1; i >= 0; i--) {
      const msg = msgs[i];
      if (msg.type === "ai") {
        const r = (msg as any).additional_kwargs?.reasoning_content;
        if (r) return r as string;
      }
    }
    return null;
  }, [stream.values?.messages]);

  // ADDED: Always-current refs so timer callbacks read live values without
  // needing them in effect dependency arrays (which would restart the timer).
  const buildDelayMessageRef = useRef<() => string>(() => "");
  buildDelayMessageRef.current = () => {
    const pending = groupedTodos.pending;
    const todoText =
      pending.length > 0 ? pending[0].content : "your request";
    const template =
      DELAY_MESSAGES[Math.floor(Math.random() * DELAY_MESSAGES.length)];
    return template.replace(/\{todo\}/g, todoText);
  };

  const getMessageCountRef = useRef<() => number>(() => 0);
  getMessageCountRef.current = () => stream.values?.messages?.length ?? 0;

  const clearDelayPanel = useCallback(() => {
    if (delayTimerRef.current) clearTimeout(delayTimerRef.current);
    delayShownRef.current = false;
    msgCountAtDelayRef.current = 0;
    setDelayMessage(null);
    setTypedText("");
  }, []);

  // ADDED: Timer effect — starts a 20-second countdown when isLoading goes
  // true. Fires the callback once (delayShownRef guard). Clears everything
  // when isLoading goes false (entire stream finished).
  useEffect(() => {
    if (isLoading) {
      if (!delayShownRef.current) {
        delayTimerRef.current = setTimeout(() => {
          delayShownRef.current = true;
          // Snapshot the message count so we can detect new arrivals later.
          msgCountAtDelayRef.current = getMessageCountRef.current();
          setDelayMessage(buildDelayMessageRef.current());
          setTypedText("");
        }, 20000);
      }
    } else {
      clearDelayPanel();
    }
    return () => {
      if (delayTimerRef.current) clearTimeout(delayTimerRef.current);
    };
  }, [isLoading, clearDelayPanel, timerEpoch]);

  // ADDED: Message-watching effect — the core of the bug fix.
  // isLoading stays true for the entire multi-step graph run, so we cannot
  // rely on it to detect individual AI messages arriving. Instead we watch
  // stream.values?.messages directly: as soon as the count grows beyond what
  // it was when the delay panel appeared, a new AI message has come in and we
  // immediately hide the panel.
  // After clearing, we bump timerEpoch so the timer effect re-runs and starts
  // a fresh 20-second countdown for any subsequent delay in the same stream.
  useEffect(() => {
    if (!delayShownRef.current) return;
    const currentCount = stream.values?.messages?.length ?? 0;
    if (currentCount > msgCountAtDelayRef.current) {
      clearDelayPanel();
      if (isLoading) setTimerEpoch((e) => e + 1);
    }
  }, [stream.values?.messages, clearDelayPanel, isLoading]);

  // ADDED: Typing animation effect — types out delayMessage one character at a
  // time (35 ms per char). Resets whenever delayMessage is replaced or cleared.
  useEffect(() => {
    if (!delayMessage) {
      setTypedText("");
      return;
    }
    let i = 0;
    typingIntervalRef.current = setInterval(() => {
      i++;
      if (i <= delayMessage.length) {
        setTypedText(delayMessage.slice(0, i));
      } else {
        if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);
      }
    }, 35);
    return () => {
      if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);
    };
  }, [delayMessage]);

  // Parse out any action requests or review configs from the interrupt
  const actionRequestsMap: Map<string, ActionRequest> | null = useMemo(() => {
    const actionRequests =
      interrupt?.value && (interrupt.value as any)["action_requests"];
    if (!actionRequests) return new Map<string, ActionRequest>();
    return new Map(actionRequests.map((ar: ActionRequest) => [ar.name, ar]));
  }, [interrupt]);

  const reviewConfigsMap: Map<string, ReviewConfig> | null = useMemo(() => {
    const reviewConfigs =
      interrupt?.value && (interrupt.value as any)["review_configs"];
    if (!reviewConfigs) return new Map<string, ReviewConfig>();
    return new Map(
      reviewConfigs.map((rc: ReviewConfig) => [rc.actionName, rc])
    );
  }, [interrupt]);

  return (
    <div className="flex flex-1 overflow-hidden">
      {hasTasks && (
        <aside className="hidden lg:flex w-56 shrink-0 flex-col overflow-y-auto border-r border-border bg-sidebar">
          <div className="border-b border-border px-4 py-3">
            <span className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
              Agent Tasks
            </span>
          </div>
          <div className="flex-1 space-y-4 px-3 py-3">
            {(
              [
                { key: "in_progress", label: "In Progress", items: groupedTodos.in_progress },
                { key: "pending",     label: "Pending",     items: groupedTodos.pending     },
                { key: "completed",   label: "Completed",   items: groupedTodos.completed   },
              ] as const
            )
              .filter(({ items }) => items.length > 0)
              .map(({ key, label, items }) => (
                <div key={key}>
                  <h3 className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-tertiary">
                    {label}
                  </h3>
                  <div className="space-y-1">
                    {items.map((todo, index) => (
                      <div
                        key={`sidebar_${key}_${todo.id ?? index}`}
                        className="flex items-start gap-2 rounded-sm px-1 py-1 text-sm"
                      >
                        {getStatusIcon(todo.status, "mt-0.5 shrink-0")}
                        <span className="break-words leading-relaxed text-inherit">
                          {todo.content}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
          </div>
        </aside>
      )}

      <div className="flex flex-1 flex-col overflow-hidden min-w-0">
      <div
        className="flex-1 overflow-y-auto overflow-x-hidden overscroll-contain"
        ref={scrollRef}
      >
        <div
          className="mx-auto w-full max-w-[1024px] px-6 pb-6 pt-4"
          ref={contentRef}
        >
          {isThreadLoading ? (
            <div className="flex items-center justify-center p-8">
              <p className="text-muted-foreground">Loading...</p>
            </div>
          ) : (
            <>
              {processedMessages.map((data, index) => {
                const messageUi = ui?.filter(
                  (u: any) => u.metadata?.message_id === data.message.id
                );
                const isLastMessage = index === processedMessages.length - 1;
                return (
                  <ChatMessage
                    key={data.message.id}
                    message={data.message}
                    toolCalls={data.toolCalls}
                    isLoading={isLoading}
                    actionRequestsMap={
                      isLastMessage ? actionRequestsMap : undefined
                    }
                    reviewConfigsMap={
                      isLastMessage ? reviewConfigsMap : undefined
                    }
                    ui={messageUi}
                    stream={stream}
                    onResumeInterrupt={resumeInterrupt}
                    graphId={assistant?.graph_id}
                  />
                );
              })}
            </>
          )}
        </div>
      </div>

      <div className="flex-shrink-0 bg-background">
        <div
          className={cn(
            "mx-4 mb-6 flex flex-shrink-0 flex-col overflow-hidden rounded-xl border border-border bg-background",
            "mx-auto w-[calc(100%-32px)] max-w-[1024px] transition-colors duration-200 ease-in-out"
          )}
        >
          {(isLoading || latestReasoning !== null || hasTasks || hasFiles) && (
            <div className="flex max-h-72 flex-col overflow-y-auto border-b border-border bg-sidebar empty:hidden">

              {/* ADDED: AI Reasoning Panel ─────────────────────────────────
                  Shown whenever the agent is loading OR has produced reasoning.
                  • Pulsing ping dot = live indicator while isLoading is true.
                  • "AI thinking..." placeholder fades in/out via animate-pulse
                    when no reasoning has arrived yet.
                  • Once reasoning arrives, the text replaces the placeholder.
                  • latestReasoning always holds the LAST AI message's content,
                    so old reasoning is automatically erased by the new one.    */}
              {(isLoading || latestReasoning) && (
                <div className="border-b border-border px-[18px] py-3">
                  {/* Header row with live indicator */}
                  <div className="mb-2 flex items-center gap-2">
                    <span className="relative flex h-2 w-2 shrink-0">
                      {isLoading && (
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75" />
                      )}
                      <span
                        className={cn(
                          "relative inline-flex h-2 w-2 rounded-full",
                          isLoading ? "bg-blue-500" : "bg-emerald-500"
                        )}
                      />
                    </span>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      AI Reasoning
                    </span>
                  </div>

                  {/* Content: placeholder while waiting, actual text once received */}
                  {latestReasoning ? (
                    <p className="text-xs leading-relaxed text-muted-foreground transition-all duration-300">
                      {latestReasoning}
                    </p>
                  ) : (
                    <p className="animate-pulse text-xs text-muted-foreground">
                      AI thinking.....
                    </p>
                  )}
                </div>
              )}
              {/* ─────────────────────────────────────────────────────────── */}

              {!metaOpen && (
                <>
                  {(() => {
                    const activeTask = todos.find(
                      (t) => t.status === "in_progress"
                    );

                    const totalTasks = todos.length;
                    const remainingTasks =
                      totalTasks - groupedTodos.pending.length;
                    const isCompleted = totalTasks === remainingTasks;

                    const tasksTrigger = (() => {
                      if (!hasTasks) return null;
                      return (
                        <button
                          type="button"
                          onClick={() =>
                            setMetaOpen((prev) =>
                              prev === "tasks" ? null : "tasks"
                            )
                          }
                          className="grid w-full cursor-pointer grid-cols-[auto_auto_1fr] items-center gap-3 px-[18px] py-3 text-left"
                          aria-expanded={metaOpen === "tasks"}
                        >
                          {(() => {
                            if (isCompleted) {
                              return [
                                <CheckCircle
                                  key="icon"
                                  size={16}
                                  className="text-success/80"
                                />,
                                <span
                                  key="label"
                                  className="ml-[1px] min-w-0 truncate text-sm"
                                >
                                  All tasks completed
                                </span>,
                              ];
                            }

                            if (activeTask != null) {
                              return [
                                <div key="icon">
                                  {getStatusIcon(activeTask.status)}
                                </div>,
                                <span
                                  key="label"
                                  className="ml-[1px] min-w-0 truncate text-sm"
                                >
                                  Task{" "}
                                  {totalTasks - groupedTodos.pending.length} of{" "}
                                  {totalTasks}
                                </span>,
                                <span
                                  key="content"
                                  className="min-w-0 gap-2 truncate text-sm text-muted-foreground"
                                >
                                  {activeTask.content}
                                </span>,
                              ];
                            }

                            return [
                              <Circle
                                key="icon"
                                size={16}
                                className="text-tertiary/70"
                              />,
                              <span
                                key="label"
                                className="ml-[1px] min-w-0 truncate text-sm"
                              >
                                Task {totalTasks - groupedTodos.pending.length}{" "}
                                of {totalTasks}
                              </span>,
                            ];
                          })()}
                        </button>
                      );
                    })();

                    const filesTrigger = (() => {
                      if (!hasFiles) return null;
                      return (
                        <button
                          type="button"
                          onClick={() =>
                            setMetaOpen((prev) =>
                              prev === "files" ? null : "files"
                            )
                          }
                          className="flex flex-shrink-0 cursor-pointer items-center gap-2 px-[18px] py-3 text-left text-sm"
                          aria-expanded={metaOpen === "files"}
                        >
                          <FileIcon size={16} />
                          Files (State)
                          <span className="h-4 min-w-4 rounded-full bg-[#2F6868] px-0.5 text-center text-[10px] leading-[16px] text-white">
                            {Object.keys(files).length}
                          </span>
                        </button>
                      );
                    })();

                    return (
                      <div className="grid grid-cols-[1fr_auto_auto] items-center">
                        {tasksTrigger}
                        {filesTrigger}
                      </div>
                    );
                  })()}
                </>
              )}

              {metaOpen && (
                <>
                  <div className="sticky top-0 flex items-stretch bg-sidebar text-sm">
                    {hasTasks && (
                      <button
                        type="button"
                        className="py-3 pr-4 first:pl-[18px] aria-expanded:font-semibold"
                        onClick={() =>
                          setMetaOpen((prev) =>
                            prev === "tasks" ? null : "tasks"
                          )
                        }
                        aria-expanded={metaOpen === "tasks"}
                      >
                        Tasks
                      </button>
                    )}
                    {hasFiles && (
                      <button
                        type="button"
                        className="inline-flex items-center gap-2 py-3 pr-4 first:pl-[18px] aria-expanded:font-semibold"
                        onClick={() =>
                          setMetaOpen((prev) =>
                            prev === "files" ? null : "files"
                          )
                        }
                        aria-expanded={metaOpen === "files"}
                      >
                        Files (State)
                        <span className="h-4 min-w-4 rounded-full bg-[#2F6868] px-0.5 text-center text-[10px] leading-[16px] text-white">
                          {Object.keys(files).length}
                        </span>
                      </button>
                    )}
                    <button
                      aria-label="Close"
                      className="flex-1"
                      onClick={() => setMetaOpen(null)}
                    />
                  </div>
                  <div
                    ref={tasksContainerRef}
                    className="px-[18px]"
                  >
                    {metaOpen === "tasks" &&
                      Object.entries(groupedTodos)
                        .filter(([_, todos]) => todos.length > 0)
                        .map(([status, todos]) => (
                          <div
                            key={status}
                            className="mb-4"
                          >
                            <h3 className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-tertiary">
                              {
                                {
                                  pending: "Pending",
                                  in_progress: "In Progress",
                                  completed: "Completed",
                                }[status]
                              }
                            </h3>
                            <div className="grid grid-cols-[auto_1fr] gap-3 rounded-sm p-1 pl-0 text-sm">
                              {todos.map((todo, index) => (
                                <Fragment key={`${status}_${todo.id}_${index}`}>
                                  {getStatusIcon(todo.status, "mt-0.5")}
                                  <span className="break-words text-inherit">
                                    {todo.content}
                                  </span>
                                </Fragment>
                              ))}
                            </div>
                          </div>
                        ))}

                    {metaOpen === "files" && (
                      <div className="mb-6">
                        <FilesPopover
                          files={files}
                          setFiles={setFiles}
                          editDisabled={
                            isLoading === true || interrupt !== undefined
                          }
                        />
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
          <form
            onSubmit={handleSubmit}
            className="flex flex-col"
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isLoading ? "Running..." : "Write your message..."}
              className="font-inherit field-sizing-content flex-1 resize-none border-0 bg-transparent px-[18px] pb-[13px] pt-[14px] text-sm leading-7 text-primary outline-none placeholder:text-tertiary"
              rows={1}
            />
            <div className="flex justify-between gap-2 p-3">
              <div className="flex justify-end gap-2">
                <Button
                  type={isLoading ? "button" : "submit"}
                  variant={isLoading ? "destructive" : "default"}
                  onClick={isLoading ? stopStream : handleSubmit}
                  disabled={!isLoading && (submitDisabled || !input.trim())}
                >
                  {isLoading ? (
                    <>
                      <Square size={14} />
                      <span>Stop</span>
                    </>
                  ) : (
                    <>
                      <ArrowUp size={18} />
                      <span>Send</span>
                    </>
                  )}
                </Button>
              </div>
            </div>
          </form>
        </div>
      </div>
      </div>

      {/* ADDED: Right-panel — only mounts when a delay message is active.
          Hidden below xl (1280 px) so it never crowds the chat on smaller
          screens. The message is typed character-by-character via typedText;
          the blinking cursor disappears once typing finishes.               */}
      {delayMessage && isLoading && (
        <aside className="hidden xl:flex w-64 shrink-0 flex-col items-center justify-center gap-5 border-l border-border bg-sidebar px-6 py-10 text-center">
          <span className="text-4xl">⏳</span>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {typedText}
            {typedText.length < delayMessage.length && (
              <span className="animate-pulse font-bold"> |</span>
            )}
          </p>
          <p className="text-[10px] uppercase tracking-widest text-tertiary/60">
            Still working…
          </p>
        </aside>
      )}
    </div>
  );
});

ChatInterface.displayName = "ChatInterface";




