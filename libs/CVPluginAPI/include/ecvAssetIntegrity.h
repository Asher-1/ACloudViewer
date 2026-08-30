// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCryptographicHash>
#include <QString>

#include "CVPluginAPI.h"
#include "aicore/asset_digests.h"

/**
 * @brief Unified integrity verification for cached remote assets (zips, GGUF
 *        models, ...).
 *
 * Design (HF-Hub / conda / Cargo pattern): verify content ONCE at ingestion,
 * record the verification in a small sidecar ledger, and answer subsequent
 * access-path checks from file stat metadata alone — no re-hashing per click.
 *
 * Concepts:
 *  - Anchor: content identity pinned in source code (algo + hex digest).
 *    SHA-256 is the default; MD5 remains available for legacy pins. An empty
 *    digest means "no content pin" (ledger/stat trust + junk checks only).
 *  - Ledger: "<file>.cvintegrity" JSON sidecar recording what was verified
 *    (algo, digest, size, mtime). Written atomically via QSaveFile.
 *  - OnMiss: policy when no usable ledger entry exists:
 *      - DeepVerify: hash the file once and re-record on match. Use for
 *        small assets (test zips) — it self-heals files downloaded before
 *        the ledger existed.
 *      - CheapChecksOnly: junk pre-checks only, never hash. Use for
 *        multi-GB models on dialog-open paths (a full pass would stall
 *        the UI for seconds).
 *
 * All functions are static and stateless; no locking is required. The ledger
 * is written with QSaveFile, so concurrent writers degrade to
 * last-writer-wins with identical content — benign.
 */
class CVPLUGIN_LIB_API ecvAssetIntegrity {
public:
    /** Content identity pinned in source code. */
    struct Anchor {
        QCryptographicHash::Algorithm algo = QCryptographicHash::Sha256;
        QByteArray digestHex;  ///< Hex digest; empty = no content pin
    };

    /** Policy when no usable ledger entry exists. */
    enum class OnMiss { CheapChecksOnly, DeepVerify };

    /** Junk pre-filter: file exists + optional size floor + GGUF magic. */
    static bool passesCheapChecks(const QString& path,
                                  qint64 minBytes = 0,
                                  bool requireGgufMagic = false);

    /**
     * One-shot full-file hash compare against the anchor. Never touches the
     * ledger. The anchor must carry a digest; with an empty digest there is
     * nothing to compare against and false is returned.
     */
    static bool verifyNow(const QString& path, const Anchor& anchor);

    /**
     * Access-path verification: a matching ledger entry (algo+digest+size+
     * mtime) trusts the file from stat metadata alone.
     * @param noLedgerExactSize legacy guard applied ONLY when no usable
     *        ledger entry exists (exact published size of a catalog model);
     *        0 disables it. Once a ledger entry exists it supersedes this
     *        guard (the entry was recorded from a byte-verified state).
     */
    static bool isVerified(const QString& path,
                           const Anchor& anchor,
                           qint64 minBytes = 0,
                           bool requireGgufMagic = false,
                           OnMiss onMiss = OnMiss::CheapChecksOnly,
                           qint64 noLedgerExactSize = 0);

    /**
     * Record a verification in the ledger. Call after the file content was
     * byte-verified (streamed digest compare at ingestion, or DeepVerify).
     * With an empty anchor digest a size-only entry is recorded (stat trust
     * without a content pin).
     */
    static bool markVerified(const QString& path, const Anchor& anchor);

    /** removeInvalidCacheFile replacement: delete path when not verified. */
    static void removeIfNotVerified(const QString& path,
                                    const Anchor& anchor,
                                    qint64 minBytes = 0,
                                    bool requireGgufMagic = false,
                                    OnMiss onMiss = OnMiss::CheapChecksOnly,
                                    qint64 noLedgerExactSize = 0);

    /** Drop the ledger entry (file deleted or replaced by the caller). */
    static void invalidate(const QString& path);

    /** Convenience: pinned SHA-256 (hex) for a published release asset file
     *  name; empty when the file is not a published asset (callers then
     *  fall back to the anchor-less size guard at ingestion). */
    static QByteArray PinnedDigest(const QString& fileName);
};
