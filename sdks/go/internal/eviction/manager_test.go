package eviction

import (
	"context"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestEvictAllWaitingConcurrentWaitReturn exercises shutdown while a durable
// wait finishes. Logging and eviction requests must not read live wait state
// without the record mutex, even after GetAllWaiting releases the cache lock.
func TestEvictAllWaitingConcurrentWaitReturn(t *testing.T) {
	logger := zerolog.New(io.Discard)
	manager := makeTestManager(nil)
	manager.l = &logger
	var workers sync.WaitGroup
	workers.Go(func() {
		for range 10000 {
			manager.RegisterRun("run/0", "step", 1, nil)
			manager.MarkWaiting("run/0", "sleep", "timer")
			manager.MarkActive("run/0")
		}
	})
	workers.Go(func() {
		for range 10000 {
			manager.EvictAllWaiting(context.Background())
		}
	})
	workers.Wait()
}

// TestEvictAllWaitingSnapshotsWaitState verifies that completing a wait during
// the eviction request cannot change its metadata or invalidate its reason.
// The callback creates the interleaving synchronously, without timing waits.
func TestEvictAllWaitingSnapshotsWaitState(t *testing.T) {
	manager := makeTestManager(nil)
	manager.RegisterRun("run/0", "step", 1, nil)
	manager.MarkWaiting("run/0", "sleep", "timer")
	manager.requestEvict = func(ctx context.Context, key string, rec *DurableRunRecord) error {
		manager.MarkActive(key)
		require.Equal(t, "sleep", rec.WaitKind)
		require.Equal(t, "timer", rec.WaitResourceID)
		require.Equal(t, 1, rec.WaitCount())
		require.NotNil(t, rec.GetWaitingSince())
		require.Equal(t, "Worker shutdown while waiting on sleep(timer)", rec.EvictionReason)
		return nil
	}

	require.Equal(t, 1, manager.EvictAllWaiting(context.Background()))
}

// TestTickSnapshotsWaitState verifies that periodic eviction uses the same
// detached metadata as shutdown. Acknowledgement can outlive a durable wait;
// finishing that wait must not mutate the record supplied to the request.
func TestTickSnapshotsWaitState(t *testing.T) {
	manager := makeTestManager(nil)
	manager.RegisterRun("run/0", "step", 1, &EvictionPolicy{TTL: time.Second})
	manager.cache.MarkWaiting("run/0", time.Now().Add(-time.Minute), "sleep", "timer")
	manager.requestEvict = func(ctx context.Context, key string, rec *DurableRunRecord) error {
		manager.MarkActive(key)
		require.Equal(t, "sleep", rec.WaitKind)
		require.Equal(t, "timer", rec.WaitResourceID)
		require.Equal(t, 1, rec.WaitCount())
		require.NotNil(t, rec.GetWaitingSince())
		require.Equal(t, "Wait TTL (1s) exceeded while waiting on sleep(timer)", rec.EvictionReason)
		return nil
	}

	require.NoError(t, manager.tick(context.Background()))
}

func makeTestManager(cancel *mockCancelLocal) *DurableEvictionManager {
	if cancel == nil {
		cancel = &mockCancelLocal{}
	}

	return &DurableEvictionManager{
		durableSlots: 10,
		cancelLocal:  cancel.call,
		requestEvict: func(ctx context.Context, key string, rec *DurableRunRecord) error {
			return nil
		},
		config: DurableEvictionConfig{CheckInterval: 1 * time.Hour},
		cache:  NewDurableEvictionCache(),
	}
}

type mockCancelLocal struct {
	calls []string
}

func (m *mockCancelLocal) call(key string) {
	m.calls = append(m.calls, key)
}

func TestHandleServerEvictionCancelsAndUnregisters(t *testing.T) {
	cancel := &mockCancelLocal{}
	mgr := makeTestManager(cancel)

	key := "run-1/0"
	mgr.RegisterRun(key, "ext-1", 2, &EvictionPolicy{TTL: 30 * time.Second})
	mgr.MarkWaiting(key, "sleep", "s1")

	mgr.HandleServerEviction("ext-1", 2)

	require.Len(t, cancel.calls, 1)
	assert.Equal(t, key, cancel.calls[0])
	assert.Nil(t, mgr.cache.Get(key))
}

func TestHandleServerEvictionUnknownIDIsNoop(t *testing.T) {
	cancel := &mockCancelLocal{}
	mgr := makeTestManager(cancel)

	mgr.RegisterRun("run-1/0", "ext-1", 1, nil)

	mgr.HandleServerEviction("no-such-id", 1)

	assert.Empty(t, cancel.calls)
	assert.NotNil(t, mgr.cache.Get("run-1/0"))
}

func TestHandleServerEvictionOnlyEvictsMatchingRun(t *testing.T) {
	cancel := &mockCancelLocal{}
	mgr := makeTestManager(cancel)

	mgr.RegisterRun("run-1/0", "ext-1", 1, &EvictionPolicy{TTL: 30 * time.Second})
	mgr.RegisterRun("run-2/0", "ext-2", 1, &EvictionPolicy{TTL: 30 * time.Second})
	mgr.MarkWaiting("run-1/0", "sleep", "s1")
	mgr.MarkWaiting("run-2/0", "sleep", "s2")

	mgr.HandleServerEviction("ext-1", 1)

	require.Len(t, cancel.calls, 1)
	assert.Equal(t, "run-1/0", cancel.calls[0])
	assert.Nil(t, mgr.cache.Get("run-1/0"))
	assert.NotNil(t, mgr.cache.Get("run-2/0"))
}

func TestHandleServerEvictionSkipsNewerInvocation(t *testing.T) {
	cancel := &mockCancelLocal{}
	mgr := makeTestManager(cancel)

	mgr.RegisterRun("run-1/0", "ext-1", 3, &EvictionPolicy{TTL: 30 * time.Second})
	mgr.MarkWaiting("run-1/0", "sleep", "s1")

	mgr.HandleServerEviction("ext-1", 2)

	assert.Empty(t, cancel.calls)
	assert.NotNil(t, mgr.cache.Get("run-1/0"))
}

func TestHandleServerEvictionEvictsExactInvocationMatch(t *testing.T) {
	cancel := &mockCancelLocal{}
	mgr := makeTestManager(cancel)

	mgr.RegisterRun("run-1/0", "ext-1", 5, &EvictionPolicy{TTL: 30 * time.Second})
	mgr.MarkWaiting("run-1/0", "sleep", "s1")

	mgr.HandleServerEviction("ext-1", 5)

	require.Len(t, cancel.calls, 1)
	assert.Equal(t, "run-1/0", cancel.calls[0])
	assert.Nil(t, mgr.cache.Get("run-1/0"))
}
