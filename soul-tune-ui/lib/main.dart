import 'package:flutter/material.dart';

import 'src/bridge.dart';
import 'src/screens/home_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initRust();
  runApp(const SoulTuneApp());
}

class SoulTuneApp extends StatelessWidget {
  const SoulTuneApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Soul-Tune',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3F51B5),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
        fontFamilyFallback: const ['Microsoft YaHei', 'PingFang SC'],
      ),
      home: const HomePage(),
    );
  }
}
