import 'package:flutter_test/flutter_test.dart';

import 'package:soul_tune_ui/main.dart';

void main() {
  testWidgets('app boots to home page', (WidgetTester tester) async {
    await tester.pumpWidget(const SoulTuneApp());
    expect(find.text('Soul-Tune 测试框架'), findsOneWidget);
  });
}
