#if canImport(Testing)
import Testing
import CandleNn

@Suite("CandleNn Swift Export Tests")
struct CandleNnExportTests {
    @Test("CandleNn swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "CandleNn swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import CandleNn

final class CandleNnExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "CandleNn swift module imported cleanly")
    }
}
#endif
