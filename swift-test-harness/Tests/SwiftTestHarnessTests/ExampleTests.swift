#if canImport(Testing)
import Testing
import CandleNn

@Suite("CandleNn Swift Export Tests")
struct CandleNnExportTests {
    @Test("CandleNn swift module imported and LayerNormConfig works")
    func testSwiftModuleLoads() throws {
        let cfg = LayerNormConfig.Companion.shared.default()
        #expect(cfg.eps == 1e-5, "LayerNormConfig default eps should be 1e-5")
        #expect(cfg.removeMean == true, "LayerNormConfig default removeMean should be true")
    }
}
#elseif canImport(XCTest)
import XCTest
import CandleNn

final class CandleNnExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        let cfg = LayerNormConfig.Companion.shared.default()
        XCTAssertEqual(cfg.eps, 1e-5, accuracy: 1e-12)
        XCTAssertTrue(cfg.removeMean)
    }
}
#endif
